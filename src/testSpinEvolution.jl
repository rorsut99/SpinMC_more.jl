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
include("evolve.jl")

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
        s9 = 0.5*Matrix(1.0I, 2, 2)

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
    a1=(1.0,0.0)
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


    uc = UnitCell(a1,a2)
    FMint = Matrix(-1.0I,dim2,dim2)      # Heisenberg interaction
    AFMint = Matrix(1.0I,dim2,dim2)      # Heisenberg interaction
    Zero = Matrix(0.0I,dim2,dim2)
    addBasisSite!(uc,(0.0,0.0),dim)

    addInteraction!(uc,gens,1,1,FMint,1,(1,0))
    addInteraction!(uc,gens,1,1,FMint,1,(0,1))

    # nearest neighbour interaction
    # added parameter to take in the order of the term in the hamiltonian

    B=[0.0,0,0.0]
    setField!(uc,gens,1,B)   

    Lsize=(10,10)       # size of lattice
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
    thermSweeps=100
    sampleSweeps=100

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
T=0.01

m,gens=runMC(T)

timeStep=0.01

evs = initEv(2, m.lattice, gens, (0,timeStep), 3)





setPhononMass!(evs, [1.0, 1.0,1.0], 3)
setPhononDamp!(evs,[0.0,0.0,0.0],3)
setStructureFactors!(evs, gens, 2)

function drive(t)
    x=0.000*cos(t)
    return (x)
end




# driveFuncs=[drive,drive]


# addPhononDrive!(evs,driveFuncs,2)







n = 2000
x = zeros(n)
x2 = zeros(n)
x3 = zeros(n)
x4 = zeros(n)
y = zeros(n)
z = zeros(n)
spinNorm = zeros(n)

phx = zeros(n)
phz = zeros(n)



# print(evs.lattice.expVals[:,5], "\n")
initPhMomentum!(evs,T,3)
evs.phononMomentaPrev = deepcopy(evs.phononMomenta)



spinEnergy, phEnergy, totalEnergy = getEvEnergy(evs,gens,evs.lattice)
print(totalEnergy/length(evs.lattice),"\n")
measureEvObservables!(evs, spinEnergy/length(evs.lattice), phEnergy/length(evs.lattice), totalEnergy/length(evs.lattice))
# print("Exp Vals Energy:",getEvSpinEnergy(evs,gens)/length(evs.lattice),"\n")
# print("State Final Energy:",getEnergy(evs.lattice,gens)/length(evs.lattice),"\n")
# print(evs.lattice.phonons[1,1])
# print(evs.lattice.expVals)
# print(evs.lattice.expVals,"\n")


tol=1e-8
evsStart=deepcopy(evs)
tols=[1e-6,1e-7,1e-8,1e-9]

for i in 1:n
    evolve!(evs, gens, 2, 3, T, 1,tol)
    s0=evs.lattice.expVals[:,1][1:3]
    # print(norm(s0),"\n")
    spinEnergy, phEnergy, totalEnergy = getEvEnergy(evs,gens,evs.lattice)
    measureEvObservables!(evs, spinEnergy/length(evs.lattice), phEnergy/length(evs.lattice), totalEnergy/length(evs.lattice))
    # print(spinEnergy/length(evs.lattice),"\n")

    x[i] = evs.lattice.expVals[1,1]



    y[i] = evs.lattice.expVals[2,1]
    z[i] = evs.lattice.expVals[3,1]
    # spinNorm[i]=sqrt(x[i]^2+y[i]^2+z[i]^2)
    # if i<(n-10)
    #     x2[i] = evs.lattice.expVals[1,1]
    #     y2[i] = evs.lattice.expVals[2,1]
    #     z2[i] = evs.lattice.expVals[3,1]
    # end

    phx[i] = evs.lattice.phonons[1,1]
    phz[i] = evs.lattice.phonons[2,1]
    # print(evs.lattice.expVals,"\n")
    if i==n
        print(totalEnergy/length(evs.lattice),"\n")
    end
end


# pop!(x2)
# pop!(y2)
# pop!(z2)
# title = string("Single Spin Evolution")
# plot(x,y,z,zlimits=(0.95,1.05),label="Spin Trajectory,t=0-1.75")
# xlabel!("Sx")
# ylabel!("Sy")
# zlabel!("Sz")


# print(evs.obs.energySeries)
tpoints=zeros(n+1)
tpoints[1]=0
for i in 2:n
    tpoints[i+1]=evs.timeStep*i
end
# tend=time()

# print(tend-tstart,"\n")

# title = string("time evolution energy")
p1=plot(tpoints,evs.obs.totalEnergySeries,label="")
xlabel!("Time(1/J)")
ylabel!("Energy Density(J/N)")


# plot(spinNorm)
# plot(x)
# plot!(x2,label=string(tols[2]))
# plot!(x3,label=string(tols[3]))
# plot!(x4,label=string(tols[4]))


p2=plot(x,y,z,label="Spin Trajectory")
xlabel!("Sx")
ylabel!("Sy")
zlabel!("Sz")

p=[p1 p2]
plot(p1,p2,layout=2)

# plot([(tpoints,evs.obs.totalEnergySeries),(x,y,z)],layout=2)



# t=LinRange(1,n,n)
# fit=maximum(phx).*exp.(-(evs.phononDamp[1].*(0.1*t)))
# plot([phx])