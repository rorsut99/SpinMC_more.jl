include("UnitCell.jl")
include("InteractionMatrix.jl")
include("Lattice.jl")

using LinearAlgebra

dim=2
a1=(1.0,0.0)
a2=(0.0,1.0)
uc = UnitCell(a1,a2)
addBasisSite!(uc,(0.0,0.0),dim)
addInteraction!(uc,1,1,Matrix(1.0I,3,3),dim,(1,0))
addInteraction!(uc,1,1,Matrix(1.0I,3,3),dim,(0,1))

Lsize=(6,6)
lattice=Lattice(uc,Lsize,dim)
