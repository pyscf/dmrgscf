DMRG interface
==============

2021-02-27

* Version 0.1

Install
-------
* Install to python site-packages folder
```
pip install git+https://github.com/pyscf/dmrgscf
```

* Install in a custom folder for development
```
git clone https://github.com/pyscf/dmrgscf /home/abc/local/path

# Set pyscf extended module path
echo 'export PYSCF_EXT_PATH=/home/abc/local/path:$PYSCF_EXT_PATH' >> ~/.bashrc
```

You can find more details of extended modules in the document
[extension modules](http://pyscf.org/install.html#extension-modules)

* Using DMRG as the FCI solver for CASSCF.  There are two DMRG solver
  interfaces implemented in this module

      Block (https://pyscf.org/Block/) or block2 (https://pypi.org/project/block2/)
      CheMPS2 (https://github.com/SebWouters/CheMPS2)

  After installing the DMRG solver, create a file dmrgscf/settings.py
  to store the path where the DMRG solver was installed.

  Detailed instruction for using ``StackBlock`` or ``block2`` as the DMRG solver for CASSCF
  can be found in this [documentation](https://block2.readthedocs.io/en/latest/user/dmrg-scf.html).
