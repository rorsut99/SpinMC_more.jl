include("UnitCell.jl")
include("InteractionMatrix.jl")
include("Lattice.jl")
include("Spin.jl")
include("Observables.jl")
include("MonteCarlo.jl")
include("Helper.jl")
include("IO.jl")
include("Phonon.jl")

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

        generators = [sx, sy, sz, s4, s5, s6, s7, s8]/sqrt(norm[1])
    end

    return generators
end

function makeLattice(dim::Int, dim2::Int, phdim::Int)

    # define cubic lattice with Heisenberg interaction
    a1=(sqrt(3)/2,1.0/2)
    a2=(sqrt(3)/2,-1.0/2)

    uc = UnitCell(a1,a2)
    FMint = Matrix(-1.0I,dim2,dim2)      # Heisenberg interaction
    AFMint = Matrix(1.0I,dim2,dim2)      # Heisenberg interaction
    Zero = Matrix(0.0I,dim2,dim2)
    addBasisSite!(uc,(0.0,0.0),dim)
    # nearest neighbour interaction
    addInteraction!(uc,1,1,FMint,dim,(1,0))
    addInteraction!(uc,1,1,FMint,dim,(0,1))
    addInteraction!(uc,1,1,FMint,dim,(1,-1))

    Lsize=(16,16)       # size of lattice
    lattice=Lattice(uc,Lsize,dim,phdim)

    generators = makeGenerators(dim)

    # push generators to lattice struct
    for gen in generators
        addGenerator!(lattice,gen,dim)
    end

    spring = [1.0, 2.0, 4.0, 5.0]
    mat = [0.0 0 0 0
           0 0.0 0 0
           0 0 0.0 0
           0.0 0 0 0
           0 0.0 0 0
           0 0 0.0 0
           0.0 0 0 0
           0 0.0 0 0]

    addSpringConstant!(lattice, spring, phdim)
    addPhononInteraction!(lattice, mat, dim, phdim)

    Sx=[0 0 1.0+0im
    0 0 1
    1 1 0]/sqrt(2)

    Sy=(-1im/sqrt(2))*[0 0 1.0+0im
    0 0 -1
    -1 1 0]

    Sz=[1.0+0im 0 0
    0 -1 0
    0 0 0]
    addSpinOperator!(lattice,Sx,3)
    addSpinOperator!(lattice,Sy,3)
    addSpinOperator!(lattice,Sz,3)
    setGenReps!(lattice,3)

    return lattice
end

function runMC(T)
    # define dimensions
    dim=3           # dimension of wavefunction (N)
    dim2=dim^2-1    # dimension of spin vector (N^2-1)

    phdim=4

    # MPI.Initialized() || MPI.Init()
    # commSize = MPI.Comm_size(MPI.COMM_WORLD)
    # commRank = MPI.Comm_rank(MPI.COMM_WORLD)

    # set sweeps
    thermSweeps=1000
    sampleSweeps=1000

    # temp=ones(length(T))
    # tmin=0.1
    # tmax=0.7
    beta=1/T
    # beta = (commSize == 1) ? 1.0/tmin : 1.0 / (reverse([ tmax * (tmin / tmax)^(n/(commSize-1)) for n in 0:commSize-1 ])[commRank+1])
    #beta = temp./T
    # T = 1000.0
    lattice = makeLattice(dim, dim2, phdim)
    lattice.Qmax = 1.0

    # run Monte Carlo sweeps
    m=MonteCarlo(lattice,beta,thermSweeps,sampleSweeps)
    run!(m,dim, phdim)
    e,e2=means(m.observables.energy)

    # # print magnetization
    # print("Magnetization vector: ", getMagnetization(m.lattice,dim), "\n")

    # # print energy
    # print("Final energy: ", e, "\nFinal energy squared: ", e2, "\n")

    
    
    return (m.energySeries)
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


# Tpoints=60
# Tvals = LinRange(0.1, 0.7, Tpoints)
# heat = zeros(Tpoints)

# runMC(Tvals)
# for i in 1:length(Tvals)
#     heat[i] = runMC(Tvals[i])
# end


# title = string("SU(", dim, ") FM heat capacity")
# plot(Tvals, heat, title=title)
# xlabel!("T")
# ylabel!("C")


energySeries=runMC(0.01)
# # plot energy vs sweeps
title = string("SU(", dim, ") FM interaction")
plot(energySeries, title=title)
xlabel!("sweeps")
ylabel!("energy density")