include("Generators.jl")
include("UnitCell.jl")
include("InteractionMatrix.jl")
include("Lattice.jl")
include("Spin.jl")
include("Observables.jl")
include("MonteCarlo.jl")
include("Helper.jl")
include("IO.jl")
include("Phonon.jl")
include("EvolveObservables.jl")
include("SphericalMidpoint.jl")

using LinearAlgebra
using Suppressor
using Plots
using MPI
using Suppressor

function makeGenerators(dim::Int)
    # produce generator matrices for different dim
    if (dim==2)
        # Pauli matrices
        sx=0.5*[0+0.0im 1.0+0im 
            1.0+0im 0+0im]
        sy=0.5*[0 -1.0im
            1.0im 0]
        sz=0.5*[1.0+0im 0
            0 -1.0+0im]
        s9 = Matrix(1.0I, 2, 2)
        # calculate norm
        norm = sx^2 + sy^2 + sz^2
        generators = [sx, sy, sz, s9]
    elseif (dim==3)
        # Gell-Mann matrices
        sx=[1+0.0im 0.0+0im  0
            0.0+0im 0+0im 0
            0 0 0]
        sy=[0 1.0 0
            0.0im 0 0
            0 0 0 ]
        sz=[0.0+0im 0 1.0
            0 0.0+0im 0
            0 0 0  ]
        s4=[0+0im 0 0.0
            1.0 0 0
            0.0 0 0]
        s5=[0 0 0.0im
            0 1.0 0
            0.0im 0 0]
        s6=[0+0.0im 0 0
            0 0 1.0
            0 0.0 0]
        s7=[0 0 0
            0 0 0.0im
            1.0 0.0im 0]
        s8=[0.0+0im 0 0
            0 0.0 0
            0 1.0 -0.0]
        # calculate norm
        norm = sx^2 + sy^2 + sz^2 + s4^2 + s5^2 + s6^2 + s7^2 + s8^2
     
        generators = [sx, sy, sz, s4, s5, s6, s7, s8]
    end
    return generators
end

function makeLattice(dim::Int, dim2::Int, phdim::Int)
    # define cubic lattice with Heisenberg interaction
    a1=(1.0,)
    a2=(0.0,1.0)
    gens=initGen(dim)
    Sx=0.5*[0+0.0im 1.0+0im 
    1.0+0im 0+0im]
    Sy=0.5*[0 -1.0im
    1.0im 0]
    Sz=0.5*[1.0+0im 0
    0 -1.0+0im]
    # Sx=[0 0 1.0+0im
    # 0 0 1
    # 1 1 0]/sqrt(2)
    # Sy=(-1im/sqrt(2))*[0 0 1.0+0im
    # 0 0 -1
    # -1 1 0]
    # Sz=[1.0+0im 0 0
    # 0 -1 0
    # 0 0 0]
    addSpinOperator!(gens,Sx)
    addSpinOperator!(gens,Sy)
    addSpinOperator!(gens,Sz)
 
    generators = makeGenerators(dim)
    # push generators to lattice struct
    for gen in generators
        addGenerator!(gens,gen)
    end
    setGenReps!(gens)
    uc = UnitCell(a1)
    FMint = Matrix(-1.0I,dim2,dim2)      # Heisenberg interaction
    AFMint = Matrix(1.0I,dim2,dim2)      # Heisenberg interaction
    Zero = Matrix(0.0I,dim2,dim2)
    addBasisSite!(uc,(0.0,),dim)
    addInteraction!(uc,gens,1,1,FMint,1,(1,))
    # addInteraction!(uc,gens,1,1,FMint,1,(0,1))
    # nearest neighbour interaction
    # added parameter to take in the order of the term in the hamiltonian
    B=[0.0,0,0.0]
    setField!(uc,gens,1,B)   
    Lsize=(100,)       # size of lattice
    lattice=Lattice(uc,Lsize,dim,phdim)
    spring = [0.0,0.0,0.0]
    # mat = [1.0; 0.0; 0.0 ;;]
    mat =-0.0*[1.0 0.0 0.0
        0.0  1.0 0.0
        0.0  0.0 1.0]
    addSpringConstant!(lattice, spring, phdim)
    addPhononInteraction!(lattice,1, gens, mat)
    return (lattice,gens)
end

function runMC(T)
    # define dimensions
    dim=2           # dimension of wavefunction (N)
    dim2=dim^2-1    # dimension of spin vector (N^2-1)
    phdim=3
    # MPI.Initialized() || MPI.Init()
    # commSize = MPI.Comm_size(MPI.COMM_WORLD)
    # commRank = MPI.Comm_rank(MPI.COMM_WORLD)
    # set sweeps
    thermSweeps=1000
    sampleSweeps=1000
    # temp=ones(length(T))
    # tmin=0.1
    # tmax=0.7
    # T=LinRange(tmin, tmax, commSize)[commRank+1]
    beta=1.0/T
    #beta=LinRange(1.0/tmax, 1.0/tmin, commSize)[commRank+1]
    # beta=1/T
    # beta = (commSize == 1) ? 1.0/tmin : 1.0 / (reverse([ tmax * (tmin / tmax)^(n/(commSize-1)) for n in 0:commSize-1 ])[commRank+1])
    # beta = temp./T
    # T = 1000.0
    lattice,gens = makeLattice(dim, dim2, phdim)
    lattice.Qmax = 0.05
    # run Monte Carlo sweeps
    m=MonteCarlo(lattice,beta,thermSweeps,sampleSweeps,dim,replicaExchangeRate=10)
    run!(m,gens,dim, phdim)
    e,e2=means(m.observables.energy)
    # # print magnetization
    # print("Magnetization vector: ", getMagnetization(m.lattice,dim), "\n")
    # # print energy
    
    print("Final Average energy: ", e, "\nFinal Average energy squared: ", e2, "\n")
    # print(m.lattice.spins)
    return (m, gens)
    # return (m.energySeries)
    # c(e) = beta * beta * (e[2] - e[1] * e[1]) * length(m.lattice) 
    # return mean(m.observables.energy, c)
end


tstart=time()
T=0.1
m,gens=runMC(T)
#test case
for site in 1:length(m.lattice)
    xi = (site-1)/99
    m.lattice.expVals[:,site] = [cos(2*pi*(xi^2))*sin(2*pi*(xi^3)), sin(2*pi*(xi^2))*sin(2*pi*(xi^3)), cos(2*pi*(xi^3)), 1.0]
end
timeStep=0.1
# finalState!(m.lattice,gens)
smp=initSphereMP(gens.dim,m.lattice,3)
setStructureFactors!(smp, gens, gens.dim)
setDT!(smp,timeStep)
function drive(t)
    x=0.000*cos(t)
    return (x)
end
# driveFuncs=[drive,drive]
# addPhononDrive!(evs,driveFuncs,2)
n = Int(500/timeStep)
x = zeros(n)
x2 = zeros(n)
x3 = zeros(n)
x4 = zeros(n)
y = zeros(n)
z = zeros(n)
spinNorm = zeros(n)
phx = zeros(n)
phz = zeros(n)
deltaE = zeros(n)
# print(evs.lattice.expVals[:,5], "\n")
spinEnergyi,a,b= getEvEnergy(smp,gens,smp.lattice)
print(spinEnergyi/length(m.lattice),"\n")
measureEvObservables!(smp, spinEnergyi/length(m.lattice), 0/length(m.lattice), 0/length(m.lattice))
# print("Exp Vals Energy:",getEvSpinEnergy(evs,gens)/length(evs.lattice),"\n")
# print("State Final Energy:",getEnergy(evs.lattice,gens)/length(evs.lattice),"\n")
# print(evs.lattice.phonons[1,1])
# print(evs.lattice.expVals)
# print(evs.lattice.expVals,"\n")
tol=1e-8
tols=[1e-6,1e-7,1e-8,1e-9]
for i in 1:n
    pos=smp.lattice.expVals[:,1]
    x[i]=pos[1]
    y[i]=pos[2]
    z[i]=pos[3]
    evolveSphereMP!(smp,smp.lattice,gens)
    spinEnergy,a,b = getEvEnergy(smp,gens,smp.lattice)
    deltaE[i] = abs(spinEnergy-spinEnergyi)
    measureEvObservables!(smp, spinEnergy/length(m.lattice), 0/length(m.lattice), 0/length(m.lattice))
    # print(spinEnergy/length(evs.lattice),"\n")
    
    if i==n
        print(spinEnergy/length(m.lattice),"\n")
    end
end
# pop!(x2)
# pop!(y2)
# pop!(z2)
# title = string("Single Spin Evolution")
# plot(x,y,label="Spin Trajectory,t=0-2")
# xlabel!("Sx")
# ylabel!("Sy")
# zlabel!("Sz")
# print(evs.obs.energySeries)
tpoints=zeros(n)
# tpoints[1]=0
for i in 1:n
    tpoints[i]=smp.dt*i
end
# tend=time()
# print(tend-tstart,"\n")
title = string("time evolution energy")
plot(tpoints,deltaE, title=title)
xlabel!("time")
ylabel!("energy")
# plot(spinNorm)
# plot(x)
# plot!(x2,label=string(tols[2]))
# plot!(x3,label=string(tols[3]))
# plot!(x4,label=string(tols[4]))
# plot(x,y,label="Spin Trajectory")
# xlabel!("Sx")
# ylabel!("Sy")
# p=[p1 p2]
# plot(p1,p2,layout=2)
# plot([(tpoints,evs.obs.totalEnergySeries),(x,y,z)],layout=2)
# t=LinRange(1,n,n)
# fit=maximum(phx).*exp.(-(evs.phononDamp[1].*(0.1*t)))
# plot([phx])










