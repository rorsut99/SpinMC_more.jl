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

function makeLattice(dim::Int, dim2::Int)

    # define cubic lattice with Heisenberg interaction
    a1=(1.0,0.0,0.0)
    a2=(0.0,1.0,0.0)
    a3=(0.0,0.0,1.0)
    uc = UnitCell(a1,a2,a3)
    FMint = Matrix(-1.0I,dim2,dim2)      # Heisenberg interaction
    AFMint = Matrix(1.0I,dim2,dim2)      # Heisenberg interaction
    addBasisSite!(uc,(0.0,0.0,0.0),dim)
    # nearest neighbour interaction
    addInteraction!(uc,1,1,AFMint,dim,(1,0,0))
    addInteraction!(uc,1,1,AFMint,dim,(0,1,0))
    addInteraction!(uc,1,1,AFMint,dim,(0,0,1))
    Lsize=(6,6,6)       # size of lattice
    lattice=Lattice(uc,Lsize,dim)

    generators = makeGenerators(dim)

    # push generators to lattice struct
    for gen in generators
        addGenerator!(lattice,gen,dim)
    end

    return lattice
end

function runMC()
    # define dimensions
    dim=2           # dimension of wavefunction (N)
    dim2=dim^2-1    # dimension of spin vector (N^2-1)
    # set sweeps
    thermSweeps=1000
    sampleSweeps=1000
    T = 1000.0
    lattice = makeLattice(dim, dim2)

    # run Monte Carlo sweeps
    m=MonteCarlo(lattice,T,thermSweeps,sampleSweeps)
    @suppress run!(m,dim)
    e,e2=means(m.observables.energy)

    # print magnetization
    print("Magnetization vector: ", getMagnetization(m.lattice,dim), "\n")

    # print energy
    print("Final energy: ", e, "\nFinal energy squared: ", e2, "\n")

    # plot energy vs sweeps
    title = string("SU(", dim, ") FM interaction")
    plot(m.energySeries, title=title)
    xlabel!("sweeps")
    ylabel!("energy density")
end

runMC()


