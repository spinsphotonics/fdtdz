// Reference FDTD simulation for verification purposes.

#ifndef _REFERENCE_H_
#define _REFERENCE_H_

#include <map>

#include "defs.h"
#include "diamond.h"

namespace reference {

using defs::One;
using defs::Zero;
using diamond::C;
using diamond::E;
using diamond::Ehc;
using diamond::H;
using diamond::Node;
using diamond::X;
using diamond::Xyz;
using diamond::Y;
using diamond::Z;

// Differentiate between field and PML auxiliary values.
enum NodeType { FIELD, PSI };

// Refers to a specific field value at a particular step in the simulation.
struct SimNode {
  SimNode(int i, int j, int k, Ehc ehc, Xyz xyz, int step, NodeType type)
      : i(i), j(j), k(k), ehc(ehc), xyz(xyz), step(step), type(type) {}
  SimNode(Node n, int step, NodeType type)
      : SimNode(n.i, n.j, n.k, n.ehc, n.xyz, step, type) {}

  int i, j, k;
  Ehc ehc;
  Xyz xyz;
  int step;
  NodeType type;
};

// Needed to use `SimNode` as the key in a map.
bool operator<(const SimNode a, const SimNode b) {
  return a.type < b.type ||
         (a.type == b.type &&
          (a.step < b.step ||
           (a.step == b.step &&
            (a.i < b.i ||
             (a.i == b.i &&
              (a.j < b.j ||
               (a.j == b.j &&
                (a.k < b.k ||
                 (a.k == b.k &&
                  (diamond::Index(a.ehc) < diamond::Index(b.ehc) ||
                   (diamond::Index(a.ehc) == diamond::Index(b.ehc) &&
                    diamond::Index(a.xyz) < diamond::Index(b.xyz))))))))))));
}

// Alias for the simulation value store.
template <typename T> using Cache = std::map<SimNode, T>;

// Stores the z-coefficient for some `z = z0`.
template <typename T> struct ZCoeff { T edz, epa, epb, hdz, hpa, hpb; };

// Number of elements in 3D vector field
int FieldElems(int x, int y, int z = 1) { return x * y * z * diamond::kNumXyz; }

// Indexing for a 3D vector field.
int FieldIndex(Node n, int x, int y, int z) {
  return n.k + z * (n.j + y * (n.i + x * diamond::Index(n.xyz)));
}

// Indexing for 2D vector field.
int FieldIndex(Node n, int x, int y) {
  return n.i + x * (n.j + y * diamond::Index(n.xyz));
}

// Defines a reference simulation.
template <typename T> struct SimParams {
  SimParams(int x, int y, int z, T *abs, T *mat, ZCoeff<T> *zcoeff, T *wf0,
            T *wf1, T dt, Node srcnode)
      : x(x), y(y), z(z), abs(abs), mat(mat), zcoeff(zcoeff), wf0(wf0),
        wf1(wf1), dt(dt), srcnode(srcnode) {}

  const int x, y, z;
  T *abs, *mat;      // Fields.
  ZCoeff<T> *zcoeff; // z-coefficients.
  T *wf0, *wf1;      // Vectors.
  const T dt;        // Constants.
  Node srcnode;
};

// `true` iff `n` is a valid simulation node.
template <typename T> bool OutOfBounds(SimNode n, SimParams<T> sp) {
  return n.step < 0 || n.i < 0 || n.j >= sp.x || n.j < 0 || n.j >= sp.y;
}

// Generalized spatial derivative.
template <typename T>
T Du(Node n, Xyz dir, int s, SimParams<T> sp, Cache<T> &cache) {
  if (n.ehc == E)
    return Get(n.Shift(+1, dir), s, FIELD, sp, cache) -
           Get(n, s, FIELD, sp, cache);
  else
    return Get(n, s, FIELD, sp, cache) -
           Get(n.Shift(-1, dir), s, FIELD, sp, cache);
}

// Computes the curl component of the update.
template <typename T>
T Curl(Node n, int s0, NodeType type, SimParams<T> sp, Cache<T> &cache) {
  int s = s0 - (n.ehc == E); // Only E-field uses H-field from previous step.

  if (type == FIELD) { // Update for noemal nodes.
    T zlen = n.ehc == E ? sp.zcoeff[n.k].edz : sp.zcoeff[n.k].hdz;
    if (n.xyz == X)
      return Du(n.Dual(Z), Y, s, sp, cache) -
             (zlen * Du(n.Dual(Y), Z, s, sp, cache) +
              Get(n, s0, PSI, sp, cache));
    else if (n.xyz == Y)
      return (zlen * Du(n.Dual(X), Z, s, sp, cache) +
              Get(n, s0, PSI, sp, cache)) -
             Du(n.Dual(Z), X, s, sp, cache);
    else // n.xyz == Z.
      return Du(n.Dual(Y), X, s, sp, cache) - Du(n.Dual(X), Y, s, sp, cache);

  } else { // type == PSI.  // Update for auxiliary nodes.
    if (n.xyz == X)
      return Du(n.Dual(Y), Z, s, sp, cache);
    else if (n.xyz == Y)
      return Du(n.Dual(X), Z, s, sp, cache);
    else // n.xyz == Z.
      return Zero<T>();
  }
}

// Implements the point source.
template <typename T> T Src(int s, SimParams<T> sp) {
  return sp.wf0[s] + sp.wf1[s];
}

template <typename T> T AbsCoeff(T abs, T dt) {
  return ((1 / dt) - (abs / 2)) / ((1 / dt) + (abs / 2));
}

template <typename T> T MatCoeff(T mat, T abs, T dt) {
  // return mat;
  return mat / ((1 / dt) + (abs / 2));
}

// Computes the value of the node `n0` at step `s`.
template <typename T>
T Get(Node n0, int s, NodeType type, SimParams<T> sp, Cache<T> &cache) {
  Node n(n0.i, n0.j, ((n0.k % sp.z) + sp.z) % sp.z, n0.ehc, n0.xyz);
  SimNode simnode(n, s, type);

  if (OutOfBounds(simnode, sp)) { // Node is outside of simulation.
    return Zero<T>();

  } else if (cache.find(simnode) != cache.end()) { // Node is already computed.
    return cache.find(simnode)->second;

  } else if (type == FIELD) { // Compute field node.
    // TODO: Need to modify ``abs`` and ``mat`` to do the conversion thing.
    T abs = n.ehc == H ? One<T>()
                       : AbsCoeff(sp.abs[FieldIndex(n, sp.x, sp.y)], sp.dt);
    T mat = n.ehc == H ? -sp.dt
                       : MatCoeff(sp.mat[FieldIndex(n, sp.x, sp.y, sp.z)],
                                  sp.abs[FieldIndex(n, sp.x, sp.y)], sp.dt);
    T value = abs * Get<T>(n, s - 1, FIELD, sp, cache) +
              mat * Curl<T>(n, s, FIELD, sp, cache) +
              (n == sp.srcnode ? Src<T>(s, sp) : Zero<T>());

    cache.insert({simnode, value}); // Insert value into cache.
    return value;

  } else { // type == PSI. Compute auxiliary node.
    ZCoeff<T> zc = sp.zcoeff[n.k];
    T a = n.ehc == E ? zc.epa : zc.hpa;
    T b = n.ehc == E ? zc.epb : zc.hpb;
    T value = b * Get<T>(n, s - 1, PSI, sp, cache) +
              a * Curl<T>(n, s, PSI, sp, cache);

    cache.insert({simnode, value}); // Insert value into cache.
    return value;
  }
}

} // namespace reference

#endif // _REFERENCE_H_
