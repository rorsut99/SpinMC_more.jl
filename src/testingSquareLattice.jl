include("UnitCell.jl")
include("InteractionMatrix.jl")
include("Lattice.jl")
include("Spin.jl")
include("Observables.jl")
include("MonteCarlo.jl")
include("Helper.jl")
include("IO.jl")

using LinearAlgebra
using Suppressor

dim=2

a1=(1.0,0.0,0.0)
a2=(0.0,1.0,0.0)
a3=(0.0,0.0,1.0)
uc = UnitCell(a1,a2,a3)
addBasisSite!(uc,(0.0,0.0,0.0),dim)
addInteraction!(uc,1,1,Matrix(1.0I,3,3),dim,(1,0,0))
addInteraction!(uc,1,1,Matrix(1.0I,3,3),dim,(0,1,0))
addInteraction!(uc,1,1,Matrix(1.0I,3,3),dim,(0,0,1))


Lsize=(6,6,6)
lattice=Lattice(uc,Lsize,dim)


sx=[0+0.0im 1.0+0im
    1.0+0im 0+0im]

sy=[0 -1.0im
    1.0im 0]

sz=[1.0+0im 0
    0 -1.0+0im]



addGenerator!(lattice,sx,dim)
addGenerator!(lattice,sy,dim)
addGenerator!(lattice,sz,dim)

thermSweeps=100
sampleSweeps=100

m=MonteCarlo(lattice,10.0,thermSweeps,sampleSweeps)
@suppress run!(m)
print(m.observables.energy)
print("\n")
e,e2=means(m.observables.energy)

print(e,e2)
print("\n")



