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
    a1=(1.0,0.0,0.0)
    a2=(0.0,1.0,0.0)
    a3=(0.0,0.0,1.0)
    uc = UnitCell(a1,a2,a3)
    FMint = Matrix(-1.0I,dim2,dim2)      # Heisenberg interaction
    AFMint = Matrix(1.0I,dim2,dim2)      # Heisenberg interaction
    Zero = Matrix(0.0I,dim2,dim2)
    addBasisSite!(uc,(0.0,0.0,0.0),dim)
    # nearest neighbour interaction
    addInteraction!(uc,1,1,FMint,dim,(1,0,0))
    addInteraction!(uc,1,1,FMint,dim,(0,1,0))
    addInteraction!(uc,1,1,FMint,dim,(0,0,1))
    Lsize=(6,6,6)       # size of lattice
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

    #addSpringConstant!(lattice, spring, phdim)
    #addPhononInteraction!(lattice, mat, dim, phdim)

    return lattice
end


MPI.Initialized() || MPI.Init()
commSize = MPI.Comm_size(MPI.COMM_WORLD)
commRank = MPI.Comm_rank(MPI.COMM_WORLD)

tmin = 0.1
tmax = 10.0
beta = (commSize == 1) ? 1.0/tmin : 1.0 / (reverse([ tmax * (tmin / tmax)^(n/(commSize-1)) for n in 0:commSize-1 ])[commRank+1])


    # define dimensions
dim=3           # dimension of wavefunction (N)
dim2=dim^2-1    # dimension of spin vector (N^2-1)

phdim=4

    # set sweeps
thermSweeps=2000
sampleSweeps=2000
T = 1000.0
lattice = makeLattice(dim, dim2, phdim)
lattice.Qmax = 1.0

    # run Monte Carlo sweeps
m=MonteCarlo(lattice,beta,thermSweeps,sampleSweeps,replicaExchangeRate=10)
run!(m,dim, phdim,outfile="simulation.h5")
e,e2=means(m.observables.energy)