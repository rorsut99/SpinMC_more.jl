include("UnitCell.jl")
include("Generators.jl")
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

function makeGenerators(dim::Int)
    # produce generator matrices for different dim
    if (dim==2)
        # Pauli matrices
        sx=[0+0.0im 1.0+0im 
            1.0+0im 0+0im]
        sy=[0 -1.0im
            1.0im 0]
        sz=[1.0+0im 0
            0 -1.0+0im]

        # calculate norm
        norm = sx^2 + sy^2 + sz^2

        generators = [sy, sz, sx]

    elseif (dim==3)
        # Gell-Mann matrices
        sx=[0+0.0im 1.0+0im  0
            1.0+0im 0+0im 0
            0 0 0]
        sy=[0 -1.0im 0
            1.0im 0 0
            0 0 0 ]
        sz=[1.0+0im 0 0
            0 -1.0+0im 0
            0 0 0  ]
        s4=[0+0im 0 1.0
            0 0 0
            1.0 0 0]
        s5=[0 0 -1.0im
            0 0 0
            1.0im 0 0]
        s6=[0+0.0im 0 0
            0 0 1.0
            0 1.0 0]
        s7=[0 0 0
            0 0 -1.0im
            0 1.0im 0]
        s8=[1.0+0im 0 0
            0 1.0 0
            0 0 -2.0]/sqrt(3)

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

    Sx=[0+0.0im 1.0+0im 
    1.0+0im 0+0im]
    Sy=[0 -1.0im
    1.0im 0]
    Sz=[1.0+0im 0
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

    addSpinOperator!(gens,Sy,2)
    addSpinOperator!(gens,Sz,2)
    addSpinOperator!(gens,Sx,2)
 

    generators = makeGenerators(dim)

    # push generators to lattice struct
    for gen in generators
        addGenerator!(gens,gen,dim)
    end
    setGenReps!(gens,2)


    uc = UnitCell(a1,a2)
    FMint = Matrix(-1.0I,dim,dim)      # Heisenberg interaction
    AFMint = Matrix(1.0I,dim,dim)      # Heisenberg interaction
    Zero = Matrix(0.0I,dim,dim)
    addBasisSite!(uc,(0.0,0.0),dim)
    Hint = [-0.5 0 0
            0 -0.5 0
            0 0 -1.0]

    addInteraction!(uc,gens,1,1,FMint,1,dim,(1,0))
    addInteraction!(uc,gens,1,1,FMint,1,dim,(0,1))
    # nearest neighbour interaction
    # added parameter to take in the order of the term in the hamiltonian

    B=[0.0,0,0.0]
    setField!(uc,gens,1,B,dim)   

    Lsize=(16,16)       # size of lattice
    lattice=Lattice(uc,Lsize,dim,phdim)

    spring = [1.0,1.0, 1.0]
    # mat = [1.0; 0.0; 0.0 ;;]
    mat =[0.0 0.0 0.0
        0.0  0.0 0.0
        0.0  0.0 0.0]

    addSpringConstant!(lattice, spring, phdim)
    addPhononInteraction!(lattice,1, gens, mat, dim, phdim)

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
    print("Final MC energy: ", e, "\nFinal energy squared: ", e2, "\n")
    print("Final energySeries energy: ", m.energySeries[end], "\n")

    # print(m.lattice.spins)
    return (m, gens)
    # return (m.energySeries)
    # c(e) = beta * beta * (e[2] - e[1] * e[1]) * length(m.lattice) 
    # return mean(m.observables.energy, c)
end

function runMD(tstep, dim, phdim)

    m,gens=runMC(1)

    evs = initEv(2, m.lattice, gens, (0,tstep), phdim)
    setPhononMass!(evs, [1.0, 1.0, 1.0], phdim)
    setPhononDamp!(evs,[0.0,0.0, 0.0],phdim)
    setStructureFactors!(evs, gens, dim)

    function drive(t)
        x=0.000*cos(t)
        return (x)
    end




    # driveFuncs=[drive,drive]


    # addPhononDrive!(evs,driveFuncs,2)







    n = Int(20/tstep)
    x = zeros(n)
    y = zeros(n)
    z = zeros(n)
    x2 = zeros(n)
    y2 = zeros(n)
    z2 = zeros(n)
    s = zeros(n)

    phx = zeros(n)
    phy = zeros(n)
    phz = zeros(n)


    T=0.1
    # print(evs.lattice.expVals[:,5], "\n")
    initPhMomentum!(evs,T,phdim)
    evs.phononMomentaPrev = deepcopy(evs.phononMomenta)

    # print(evs.lattice.phonons[1,1])
    # print(evs.lattice.expVals)
    spinEnerg, phEnerg, totalEnerg = getEvEnergy(evs,gens)
    print("Initial MD energy: ", spinEnerg/length(evs.lattice), "\n")


    for i in 1:n
        for j in 1:length(evs.lattice)
            x[i] += evs.lattice.expVals[1,j]
            y[i] += evs.lattice.expVals[2,j]
            z[i] += evs.lattice.expVals[3,j]
        end
        x[i] = x[i]/(length(evs.lattice))
        y[i] = y[i]/(length(evs.lattice))
        z[i] = z[i]/(length(evs.lattice))

        # x[i] += evs.lattice.expVals[1,32]
        # y[i] += evs.lattice.expVals[2,32]
        # z[i] += evs.lattice.expVals[3,32]

        # x2[i] += evs.lattice.expVals[1,100]
        # y2[i] += evs.lattice.expVals[2,100]
        # z2[i] += evs.lattice.expVals[3,100]

        s[i] = sqrt(x[i]^2 + y[i]^2 + z[i]^2)

        evolve!(evs, gens, 2, 3, T, 1)
        spinEnergy, phEnergy, totalEnergy = getEvEnergy(evs,gens)
        measureEvObservables!(evs, spinEnergy/length(evs.lattice), phEnergy/length(evs.lattice), totalEnergy/length(evs.lattice))

        # phx[i] = evs.lattice.phonons[1,1]
        # phy[i] = evs.lattice.phonons[2,1]
        # phz[i] = evs.lattice.phonons[3,1]
    end

    return evs, s

end

# print(evs.lattice.expVals)
# title = string("time evolution")
# p1 = plot(x, y, z, title=title)
# p2 = plot(x2, y2, z2, title=title)
# plot(p1, p2, layout = 2)
# xlabel!("Sx")
# ylabel!("Sy")

# plot([x,y,z], title="spin components")

# print(evs.obs.energySeries)

stepSizes = [0.1, 0.05, 0.01, 0.005, 0.001]
t1 = Vector(LinRange(0, 20, Int(20/stepSizes[1])))
t2 = Vector(LinRange(0, 20, Int(20/stepSizes[2])))
t3 = Vector(LinRange(0, 20, Int(20/stepSizes[3])))
t4 = Vector(LinRange(0, 20, Int(20/stepSizes[4])))
t5 = Vector(LinRange(0, 20, Int(20/stepSizes[5])))
e1, s1 = runMD(stepSizes[1], 2, 3)
e2, s2 = runMD(stepSizes[2], 2, 3)
e3, s3 = runMD(stepSizes[3], 2, 3)
# e4, s4 = runMD(stepSizes[4], 2, 3)
# e5, s5 = runMD(stepSizes[5], 2, 3)
plot(t1, e1.obs.spinEnergySeries, label=string(stepSizes[1]), title="time evolution energy")
plot!(t2, e2.obs.spinEnergySeries, label=string(stepSizes[2]))
plot!(t3, e3.obs.spinEnergySeries, label=string(stepSizes[3]))
# plot!(t4, e4.obs.spinEnergySeries, label=string(stepSizes[4]))
# plot!(t5, e5.obs.spinEnergySeries, label=string(stepSizes[5]))
xlabel!("time")
ylabel!("energy")


# plot(t1, s1, label=string(stepSizes[1]), title="time evolution magnetization")
# plot!(t2, s2, label=string(stepSizes[2]))
# plot!(t3, s3, label=string(stepSizes[3]))
# plot!(t4, s4, label=string(stepSizes[4]))
# plot!(t5, s5, label=string(stepSizes[5]))
# xlabel!("time")
# ylabel!("magnetization")
# plot(t2, runMD(stepSizes[2], 2, 3))
# p3 = plot(runMD(stepSizes[3], 2, 3))
# p4 = plot(runMD(stepSizes[4], 2, 3))


# title = string("time evolution energy")
# plot(p1, title=title)
# xlabel!("time")
# ylabel!("energy")

# t=LinRange(1,n,n)
# fit=maximum(phx).*exp.(-(evs.phononDamp[1].*(0.1*t)))
# plot(0.1*t,[phx, phy, phz],title=title)



# evsObj, mag = runMD(0.0005, 2, 3)
