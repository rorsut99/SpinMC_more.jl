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

        generators = [sx, sy, sz]/sqrt(norm[1])

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
    a1=(sqrt(3)/2,1.0/2)
    a2=(sqrt(3)/2,-1.0/2)

    gens=initGen(dim)

    # Sx=[0+0.0im 1.0+0im 
    # 1.0+0im 0+0im]
    # Sy=[0 -1.0im
    # 1.0im 0]
    # Sz=[1.0+0im 0
    # 0 -1.0+0im]

    Sx=[0 0 1.0+0im
    0 0 1
    1 1 0]/sqrt(2)

    Sy=(-1im/sqrt(2))*[0 0 1.0+0im
    0 0 -1
    -1 1 0]

    Sz=[1.0+0im 0 0
    0 -1 0
    0 0 0]

    addSpinOperator!(gens,Sy,3)
    addSpinOperator!(gens,Sz,3)
    addSpinOperator!(gens,Sx,3)
 

    generators = makeGenerators(dim)

    # push generators to lattice struct
    for gen in generators
        addGenerator!(gens,gen,dim)
    end
    setGenReps!(gens,3)


    uc = UnitCell(a1,a2)
    FMint = Matrix(-1.50I,dim,dim)      # Heisenberg interaction
    AFMint = Matrix(1.0I,dim,dim)      # Heisenberg interaction
    Zero = Matrix(0.0I,dim,dim)
    addBasisSite!(uc,(0.0,0.0),dim)
    # nearest neighbour interaction
    # added parameter to take in the order of the term in the hamiltonian
    # bilinear interaction
    addInteraction!(uc,gens,1,1,AFMint,1,dim,(1,0))
    addInteraction!(uc,gens,1,1,AFMint,1,dim,(0,1))
    addInteraction!(uc,gens,1,1,AFMint,1,dim,(1,-1))

    B=[1.0,0,0]
    setField!(uc,gens,1,B,dim)

    # # biquadratic interaction
    addInteraction!(uc,gens,1,1,FMint,2,dim,(1,0))
    addInteraction!(uc,gens,1,1,FMint,2,dim,(0,1))
    addInteraction!(uc,gens,1,1,FMint,2,dim,(1,-1))

    

    Lsize=(16,16)       # size of lattice
    lattice=Lattice(uc,Lsize,dim,phdim)

    

    spring = [1.0,1.0]
    # mat = [1.0; 0.0; 0.0 ;;]
    mat =[1.0 0.0
        0.0  0.0
        0.0  1.0]

    addSpringConstant!(lattice, spring, phdim)
    addPhononInteraction!(lattice,1, gens, mat, dim, phdim)

    

    # Sx=[0 0 1.0+0im
    # 0 0 1
    # 1 1 0]/sqrt(2)

    # Sy=(-1im/sqrt(2))*[0 0 1.0+0im
    # 0 0 -1
    # -1 1 0]

    # Sz=[1.0+0im 0 0
    # 0 -1 0
    # 0 0 0]

    return (lattice,gens)
end

function runMC(T)
    # define dimensions
    dim=3           # dimension of wavefunction (N)
    dim2=dim^2-1    # dimension of spin vector (N^2-1)

    phdim=2

    # MPI.Initialized() || MPI.Init()
    # commSize = MPI.Comm_size(MPI.COMM_WORLD)
    # commRank = MPI.Comm_rank(MPI.COMM_WORLD)

    # set sweeps
    thermSweeps=10
    sampleSweeps=10

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
    # print("Final energy: ", e, "\nFinal energy squared: ", e2, "\n")

    # print(m.lattice.spins)
    return (m, gens)
    # return (m.energySeries)
    # c(e) = beta * beta * (e[2] - e[1] * e[1]) * length(m.lattice) 
    # return mean(m.observables.energy, c)
end

# Tvals = LinRange(0.1, 8, 40)
# heat = zeros(40)
# for i in 1:length(Tvals)
#     heat[i] = runMC(Tvals[i])
# end

#  # plot energy vs sweeps
# title = string("SU(", dim, ") FM heat capacity")
# plot(Tvals, heat, title=title)
# xlabel!("T")
# ylabel!("C")


# Tpoints=30
# Tvals = LinRange(0.1, 0.7, Tpoints)
# heat = zeros(Tpoints)

m,gens=runMC(0.01)


# for i in 1:length(Tvals)
#     heat[i] = runMC(Tvals[i])
# end


# title = string("SU(", dim, ") FM heat capacity")
# plot(Tvals, heat, title=title)
# xlabel!("T")
# ylabel!("C")

# dim=2        # dimension of wavefunction (N)
# dim2=dim^2-1    # dimension of spin vector (N^2-1)

# phdim=4

# lattice,generators=makeLattice(dim,dim2,phdim)

# m=runMC(0.01)
# print(m.energySeries)
# # plot energy vs sweeps
# title = string("SU(", dim, ") FM interaction")
# plot(m.energySeries, title=title)
# xlabel!("sweeps")
# ylabel!("energy density")