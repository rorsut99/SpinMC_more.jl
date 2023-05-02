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
using Plots

dim=3
dim2=dim^2-1


a1=(1.0,0.0,0.0)
a2=(0.0,1.0,0.0)
a3=(0.0,0.0,1.0)
uc = UnitCell(a1,a2,a3)
addBasisSite!(uc,(0.0,0.0,0.0),dim)
addInteraction!(uc,1,1,Matrix(-1.0I,dim2,dim2),dim,(1,0,0))
addInteraction!(uc,1,1,Matrix(-1.0I,dim2,dim2),dim,(0,1,0))
addInteraction!(uc,1,1,Matrix(-1.0I,dim2,dim2),dim,(0,0,1))


Lsize=(6,6,6)
lattice=Lattice(uc,Lsize,dim)


norm3=sqrt(4)
norm2=sqrt(3)

# sx=[0+0.0im 1.0+0im 
#     1.0+0im 0+0im]/norm2

# sy=[0 -1.0im
#     1.0im 0 ]/norm2

# sz=[1.0+0im 0
#     0 -1.0+0im  ]/norm2

sx=[0+0.0im 1.0+0im  0
    1.0+0im 0+0im 0
    0 0 0]/(norm3)

sy=[0 -1.0im 0
    1.0im 0 0
    0 0 0 ]/(norm3)

sz=[1.0+0im 0 0
    0 -1.0+0im 0
    0 0 0  ]/(norm3)

s4=[0+0im 0 1.0
0 0 0
1.0 0 0]/(norm3)

s5=[0 0 -1.0im
0 0 0
1.0im 0 0]/(norm3)

s6=[0+0.0im 0 0
0 0 1.0
0  1.0 0]/(norm3)

s7=[0 0 0
0 0 -1.0im
0 1.0im 0]/(norm3)

s8=[1.0+0im 0 0
0 1.0 0
0 0 -2.0]/(sqrt(3)*norm3)

# sum=sx^2+sy^2+sz^2
# print(sum)

addGenerator!(lattice,sx,dim)
addGenerator!(lattice,sy,dim)
addGenerator!(lattice,sz,dim)
addGenerator!(lattice,s4,dim)
addGenerator!(lattice,s5,dim)
addGenerator!(lattice,s6,dim)
addGenerator!(lattice,s7,dim)
addGenerator!(lattice,s8,dim)



thermSweeps=1000
sampleSweeps=1000


m=MonteCarlo(lattice,100.0,thermSweeps,sampleSweeps)
@suppress run!(m)
print(m.observables.energy)
print("\n")
e,e2=means(m.observables.energy)

print("\n")
print(getMagnetization(m.lattice,dim))



print(e,e2)
print("\n")

plot(m.energySeries)



