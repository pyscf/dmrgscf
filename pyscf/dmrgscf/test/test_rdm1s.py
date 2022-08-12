#!/usr/bin/env python
# Copyright 2022 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import unittest
import numpy
from pyscf import gto, scf, mcscf, dmrgscf


# Perform CASCI calculation, return spin-1-RDM for the specified root.
def casci_rdm1s(mf, norb, nelec, nroots=1, root=0):
    mc = mcscf.CASCI(mf, norb, nelec)
    mc.fcisolver.conv_tol = 1e-12
    mc.fcisolver.nroots = nroots
    mc.fix_spin(shift=10.0)
    mc.kernel()
    civec = mc.fcisolver.ci[root] if nroots > 1 else mc.fcisolver.ci
    rdm1s = mc.fcisolver.make_rdm1s(civec, mc.fcisolver.norb, mc.fcisolver.nelec)
    return numpy.array(rdm1s)


# Perform DMRG-CASCI calculation, return spin-1-RDM for the specified root.
def dmrgci_rdm1s(mf, norb, nelec, nroots=1, root=0):
    mc = mcscf.CASCI(mf, norb, nelec)
    mc.fcisolver = dmrgscf.DMRGCI(mf.mol, maxM=500, tol=1e-10)
    mc.fcisolver.nroots = nroots
    mc.kernel()
    rdm1s = dmrgscf.DMRGCI.make_rdm1s(mc.fcisolver, root, norb, nelec)
    return numpy.array(rdm1s)


# Test cases for DMRGCI.make_rdm1s (reduced one-particle density matrix with spin components)
class KnownValues(unittest.TestCase):

    def test_make_rdm1s_Li(self):
        # Test case Li atom (doublet): ROHF reference orbitals.
        mol = gto.Mole(atom='Li 0 0 0', basis='def2-SVP', spin=1).build()
        mf = scf.ROHF(mol).run()

        # Calculate 1-RDM for three electrons in five orbitals with CASCI.
        rdm1s_casci = casci_rdm1s(mf, 5, 3, nroots=2, root=0)

        # Calculate 1-RDM for three electrons in five orbitals with DMRG-CASCI.
        rdm1s_dmrgci = dmrgci_rdm1s(mf, 5, 3, nroots=2, root=0)

        # Main check: 1-RDMs from normal FCI and DMRG must be sufficiently close.
        self.assertTrue(numpy.allclose(rdm1s_dmrgci, rdm1s_casci, atol=1.0e-5, rtol=0.0))

        # Sanity check: correct number of electrons.
        self.assertAlmostEqual(numpy.trace(rdm1s_dmrgci[0]), 2.0, delta=1e-12)
        self.assertAlmostEqual(numpy.trace(rdm1s_dmrgci[1]), 1.0, delta=1e-12)

        # Sanity check: density matrix is symmetric.
        self.assertTrue(numpy.allclose(rdm1s_dmrgci[0], rdm1s_dmrgci[0].T, atol=1e-12, rtol=0.0))
        self.assertTrue(numpy.allclose(rdm1s_dmrgci[1], rdm1s_dmrgci[1].T, atol=1e-12, rtol=0.0))

    def test_make_rdm1s_Be(self):
        # Test case Be atom (singlet): RHF reference orbitals.
        mol = gto.Mole(atom='Be 0 0 0', basis='def2-SVP', spin=0).build()
        mf = scf.RHF(mol).run()

        # Calculate 1-RDM for four electrons in six orbitals with CASCI. (1s2s2p3s)
        rdm1s_casci = casci_rdm1s(mf, 6, 4)

        # Calculate 1-RDM for four electrons in six orbitals with DMRG-CASCI.
        rdm1s_dmrgci = dmrgci_rdm1s(mf, 6, 4)

        # Main check: 1-RDMs from normal FCI and DMRG must be sufficiently close.
        self.assertTrue(numpy.allclose(rdm1s_dmrgci, rdm1s_casci, atol=1.0e-5, rtol=0.0))

        # Sanity check: correct number of electrons.
        self.assertAlmostEqual(numpy.trace(rdm1s_dmrgci[0]), 2.0, delta=1e-12)
        self.assertAlmostEqual(numpy.trace(rdm1s_dmrgci[1]), 2.0, delta=1e-12)

        # Sanity check: density matrix is symmetric.
        self.assertTrue(numpy.allclose(rdm1s_dmrgci[0], rdm1s_dmrgci[0].T, atol=1e-12, rtol=0.0))
        self.assertTrue(numpy.allclose(rdm1s_dmrgci[1], rdm1s_dmrgci[1].T, atol=1e-12, rtol=0.0))

    def test_make_rdm1s_N(self):
        # Test case N atom (quartet): ROHF reference orbitals.
        mol = gto.Mole(atom='N 0 0 0', basis='def2-SVP', spin=3).build()
        mf = scf.ROHF(mol)
        mf.kernel()

        # Calculate 1-RDM for seven electrons in nine orbitals with CASCI. (1s2s2p3s3p)
        rdm1s_casci = casci_rdm1s(mf, 9, 7)

        # Calculate 1-RDM for seven electrons in nine orbitals with DMRG-CASCI.
        rdm1s_dmrgci = dmrgci_rdm1s(mf,9, 7)

        # Main check: 1-RDMs from normal FCI and DMRG must be sufficiently close.
        self.assertTrue(numpy.allclose(rdm1s_dmrgci, rdm1s_casci, atol=1.0e-5, rtol=0.0))

        # Sanity check: correct number of electrons.
        self.assertAlmostEqual(numpy.trace(rdm1s_dmrgci[0]), 5.0, delta=1e-12)
        self.assertAlmostEqual(numpy.trace(rdm1s_dmrgci[1]), 2.0, delta=1e-12)

        # Sanity check: density matrix is symmetric.
        self.assertTrue(numpy.allclose(rdm1s_dmrgci[0], rdm1s_dmrgci[0].T, atol=1e-12, rtol=0.0))
        self.assertTrue(numpy.allclose(rdm1s_dmrgci[1], rdm1s_dmrgci[1].T, atol=1e-12, rtol=0.0))

    def test_make_rdm1s_F_exc(self):
        # Test case F atom (doublet): ROHF reference orbitals.
        mol = gto.Mole(atom='F 0 0 0', basis='def2-SVP', spin=1).build()
        mf = scf.ROHF(mol)
        mf.kernel()

        # Checking if the rdm1s calculation also works for excited states.
        # CASCI calculation with seven electrons in four orbitals (2s2p).
        # Among the states that can be calculated with this particular active space, calculate
        # the density matrix for state with root == 3. It is non-degenerate.
        rdm1s_casci = casci_rdm1s(mf, 4, 7, nroots=4, root=3)

        # Calculate 1-RDM for the same state with DMRG-CASCI.
        rdm1s_dmrgci = dmrgci_rdm1s(mf, 4, 7, nroots=4, root=3)

        # Main check: 1-RDMs from normal FCI and DMRG must be sufficiently close.
        self.assertTrue(numpy.allclose(rdm1s_dmrgci, rdm1s_casci, atol=1.0e-5, rtol=0.0))

        # Sanity check: correct number of electrons.
        self.assertAlmostEqual(numpy.trace(rdm1s_dmrgci[0]), 4.0, delta=1e-12)
        self.assertAlmostEqual(numpy.trace(rdm1s_dmrgci[1]), 3.0, delta=1e-12)

        # Sanity check: density matrix is symmetric.
        self.assertTrue(numpy.allclose(rdm1s_dmrgci[0], rdm1s_dmrgci[0].T, atol=1e-12, rtol=0.0))
        self.assertTrue(numpy.allclose(rdm1s_dmrgci[1], rdm1s_dmrgci[1].T, atol=1e-12, rtol=0.0))


if __name__ == '__main__':
    unittest.main()
