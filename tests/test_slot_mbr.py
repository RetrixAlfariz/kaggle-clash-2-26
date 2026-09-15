import itertools
import bisect
import math
import unittest

import numpy as np

from src.slot_mbr import decode


def compatible(pairs, ids):
    chosen = sorted((pairs[i] for i in ids), key=lambda x: (x[1], x[0]))
    return all(chosen[i][1] <= chosen[i + 1][0] for i in range(len(chosen) - 1))


def exhaustive(pairs, energies, k):
    m, labels = energies.shape
    worlds = []
    for ids in itertools.combinations(range(m), k):
        if compatible(pairs, ids):
            for ls in itertools.product(range(labels), repeat=k):
                ordered = sorted(zip(ids, ls), key=lambda x: (pairs[x[0]][1], pairs[x[0]][0]))
                worlds.append((ordered, sum(energies[i, l] for i, l in ordered)))
    vals = np.array([x[1] for x in worlds])
    z = float(vals.max() + np.log(np.exp(vals - vals.max()).sum()))
    probs = np.exp(vals - z)
    mu = np.zeros((k,m,labels))
    for world, pr in zip(worlds, probs):
        for j,(i,l) in enumerate(world[0]):mu[j,i,l]+=pr
    # Expected exact-slot utility for every decision.
    best = (-1.0, None)
    for world,_ in worlds:
        decision=tuple(world)
        u = sum(mu[j,i,l] for j,(i,l) in enumerate(decision))
        out = (u, decision)
        if u > best[0] + 1e-12 or (abs(u - best[0]) <= 1e-12 and tuple(pairs[i][0] for i, _ in decision) < tuple(pairs[i][0] for i, _ in best[1])):
            best = out
    map_world = max(worlds, key=lambda x: x[1])
    return z, best[0], best[1], float(map_world[1]), tuple(map_world[0]),mu


def labeled_reference(pairs, energies, k, t):
    """Independent expanded labeled-candidate prefix/suffix DP, no token grouping."""
    m, labels = energies.shape
    candidates=sorted([(i,l) for i in range(m) for l in range(labels)],key=lambda x:(pairs[x[0],1],pairs[x[0],0],x[1]))
    ends=[int(pairs[i,1]) for i,l in candidates];n=len(candidates)
    f=np.full((n+1,k+1),-np.inf);f[:,0]=0
    previous=[]
    for j,(i,l) in enumerate(candidates):
        p=bisect.bisect_right(ends,int(pairs[i,0]),hi=j);previous.append(p)
        f[j+1,1:]=np.logaddexp(f[j,1:],f[p,:-1]+energies[i,l])
    z=float(f[n,k])
    reverse=sorted(candidates,key=lambda x:(pairs[x[0],0],pairs[x[0],1],x[1]))
    starts=[int(pairs[i,0]) for i,l in reverse]
    bwd=np.full((n+1,k+1),-np.inf);bwd[:,0]=0
    for j in range(n-1,-1,-1):
        i,l=reverse[j];following=bisect.bisect_left(starts,int(pairs[i,1]),lo=j+1)
        bwd[j,1:]=np.logaddexp(bwd[j+1,1:],energies[i,l]+bwd[following,:-1])
    np.testing.assert_allclose(z,bwd[0,k],rtol=0,atol=1e-10)
    mu = np.zeros((k, m, labels))
    for j,(i,l) in enumerate(candidates):
        following=bisect.bisect_left(starts,int(pairs[i,1]))
        mu[:,i,l]=np.exp(f[previous[j],:k]+energies[i,l]+bwd[following,k-1::-1]-z)
    return z, mu


class SlotMBRTests(unittest.TestCase):
    def test_protocol_fixture(self):
        pairs = np.array([[0, 1], [1, 2], [2, 3], [0, 2], [1, 3]], dtype=np.int64)
        energies = np.array([[.5, 1], [1.5, -1.5], [-.5, .5], [-1, 0], [-.5, -1]], dtype=np.float64)
        result, d = decode(3, pairs, energies, 2,return_marginals=True)
        self.assertEqual(result, [(0, 1, 1), (1, 2, 0)])
        self.assertAlmostEqual(d["expected_utility"], .8509544260, places=8)
        collapsed,_=decode(3,pairs,energies.max(1)[:,None],2)
        self.assertEqual([x[:2] for x in collapsed],[(0,1),(2,3)])
        mu=np.asarray(d['slot_marginals'])
        self.assertAlmostEqual(mu[0,0,1]+mu[1,2,1],.8021515788,places=8)

    def test_randomized_against_independent_enumeration(self):
        rng = np.random.default_rng(20260915)
        for _ in range(1000):
            t = int(rng.integers(1, 7))
            w = int(rng.integers(1, min(3, t) + 1))
            pairs = np.array([(a, b) for a in range(t) for b in range(a + 1, min(t, a + w) + 1)], dtype=np.int64)
            labels = int(rng.integers(1, 4))
            energies = rng.normal(0, 2, size=(len(pairs), labels))
            k = int(rng.integers(1, min(3, t) + 1))
            z, utility, decision, map_score, map_decision,brute_mu = exhaustive(pairs, energies, k)
            ref_z, ref_mu = labeled_reference(pairs, energies, k, t)
            result, d = decode(t, pairs, energies, k, return_marginals=True)
            got = tuple((next(i for i, p in enumerate(pairs) if tuple(p) == x[:2]), x[2]) for x in result)
            self.assertAlmostEqual(d["logZ"], z, places=9)
            self.assertAlmostEqual(d["logZ"], ref_z, places=9)
            np.testing.assert_allclose(d['slot_marginals'],ref_mu,rtol=0,atol=2e-10)
            np.testing.assert_allclose(d['slot_marginals'],brute_mu,rtol=0,atol=2e-10)
            self.assertAlmostEqual(d['map_score'],map_score,places=9)
            map_got=tuple((next(i for i,p in enumerate(pairs) if tuple(p)==x[:2]),x[2]) for x in d['map_prediction'])
            self.assertEqual(map_got,map_decision)
            self.assertAlmostEqual(d["expected_utility"], utility, places=9)
            self.assertEqual(got, decision)
            self.assertTrue(np.allclose(np.asarray(d["slot_marginals"]).sum(axis=(1, 2)), 1.0, atol=2e-10))

    def test_edges_and_failures(self):
        empty = np.empty((0, 2), dtype=np.int64)
        en = np.empty((0, 2), dtype=np.float64)
        self.assertEqual(decode(0, empty, en, 0)[0], [])
        pairs = np.array([[0, 1], [1, 2]], dtype=np.int64)
        energies = np.zeros((2, 1))
        self.assertEqual(decode(2, pairs, energies, 0)[0], [])
        with self.assertRaises(ValueError): decode(2, pairs, energies, 3)
        with self.assertRaises(ValueError): decode(2, np.array([[0, 2], [0, 2]]), np.zeros((2, 1)), 1)
        with self.assertRaises(ValueError): decode(2, np.array([[0, 3]]), np.zeros((1, 1)), 1)
        with self.assertRaises(ValueError): decode(3,np.array([[0,2],[1,3]]),np.zeros((2,1)),2)
        self.assertEqual(decode(2,pairs,np.array([[50.,-50.],[-50.,50.]]),2)[0],[(0,1,0),(1,2,1)])
        self.assertEqual(decode(5,np.array([[0,1]]),np.zeros((1,1)),1)[0],[(0,1,0)])

    def test_extreme_uniform_and_map_mbr_difference(self):
        pairs=np.array([[0,1],[1,2],[2,3],[0,2],[1,3]])
        for energies in (np.zeros((5,2)),np.full((5,2),1e-10),
                         np.array([[50,-50],[-50,50],[-50,-50],[-50,-50],[-50,-50]],dtype=float)):
            z,u,_,ms,_,mu=exhaustive(pairs,energies,2)
            _,d=decode(3,pairs,energies,2,return_marginals=True)
            self.assertAlmostEqual(z,d['logZ'],places=9)
            self.assertAlmostEqual(u,d['expected_utility'],places=9)
            self.assertAlmostEqual(ms,d['map_score'],places=9)
            np.testing.assert_allclose(mu,d['slot_marginals'],rtol=0,atol=2e-10)
        e=np.array([-1.0677,-1.0765,-1.1249,-.1616,-.8409])[:,None]
        result,d=decode(3,pairs,e,2)
        self.assertEqual(result,[(0,1,0),(2,3,0)])
        self.assertEqual(d['map_prediction'],[(0,2,0),(2,3,0)])
        self.assertGreater(d['mbr_expected_utility'],d['map_expected_utility'])

    def test_ties_prefer_skip_then_smaller_start(self):
        pairs = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int64)
        result, _ = decode(2, pairs, np.zeros((3, 1)), 1)
        self.assertEqual(result, [(0, 1, 0)])


if __name__ == "__main__":
    unittest.main()
