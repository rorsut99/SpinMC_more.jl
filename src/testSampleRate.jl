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
include("SchrodingerMidpoint.jl")
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


function makeLattice(dim::Int,  phdim::Int)
    a1=(1/sqrt(2),0,1/sqrt(2))
    a2=(1/sqrt(2),1/sqrt(2),0)
    a3=(0,1/sqrt(2),1/sqrt(2))

    gens=initGen(dim)

    Sx=0.5*[0+0.0im 1.0+0im 
    1.0+0im 0+0im]
    Sy=0.5*[0 -1.0im
    1.0im 0]
    Sz=0.5*[1.0+0im 0
    0 -1.0+0im]

    addSpinOperator!(gens,Sx)
    addSpinOperator!(gens,Sy)
    addSpinOperator!(gens,Sz)

    generators = makeGenerators(dim)

    for gen in generators
        addGenerator!(gens,gen)
    end
    setGenReps!(gens)

    uc = UnitCell(a1,a2,a3)
    



    K0=-1.0
    K1=0.1
    K2=0.05

    Phi=0.0
    xyInteraction=1.0*[K1*cos(Phi)^2+K2*sin(Phi)^2 0.0 (K1-K2)*sin(Phi)*cos(Phi)
    0 K0 0
    (K1-K2)*sin(Phi)*cos(Phi) 0 K2*cos(Phi)^2+K1*sin(Phi)^2 ]

    Phi=4pi/3
    xzInteraction=1.0*[K1*cos(Phi)^2+K2*sin(Phi)^2 0.0 (K1-K2)*sin(Phi)*cos(Phi)
    0 K0 0
    (K1-K2)*sin(Phi)*cos(Phi) 0 K2*cos(Phi)^2+K1*sin(Phi)^2 ]


    Phi=2pi/3
    yzInteraction=1.0*[K1*cos(Phi)^2+K2*sin(Phi)^2 0.0 (K1-K2)*sin(Phi)*cos(Phi)
    0 K0 0
    (K1-K2)*sin(Phi)*cos(Phi) 0 K2*cos(Phi)^2+K1*sin(Phi)^2 ]


    addBasisSite!(uc,(0.0,0.0,0.0),dim)


    addInteraction!(uc,gens,1,1,xyInteraction,1,(0,1,0))
    addInteraction!(uc,gens,1,1,xyInteraction,1,(-1,0,1))

    addInteraction!(uc,gens,1,1,xzInteraction,1,(1,0,0))
    addInteraction!(uc,gens,1,1,xzInteraction,1,(0,1,-1))
    
    addInteraction!(uc,gens,1,1,yzInteraction,1,(0,0,1))
    addInteraction!(uc,gens,1,1,yzInteraction,1,(-1,1,0))

    Lsize=(6, 6, 6)
    lattice=Lattice(uc,Lsize,dim,phdim) 



    spring = [1.0,1.0]
    mat =-1.0*[1.0  0.0
               0.0  0.0
               0.0  1.0]
    addSpringConstant!(lattice, spring, phdim)
    addPhononInteraction!(lattice,1, gens, mat)
    return (lattice,gens)
end


function runMC(T)
    # define dimensions
    dim=2           # dimension of wavefunction (N)
    dim2=dim^2-1    # dimension of spin vector (N^2-1)
    phdim=2
    # MPI.Initialized() || MPI.Init()
    # commSize = MPI.Comm_size(MPI.COMM_WORLD)
    # commRank = MPI.Comm_rank(MPI.COMM_WORLD)
    # set sweeps
    thermSweeps=300
    sampleSweeps=700


    # tmin=1.0
    # tmax=8.0
    # T=LinRange(tmin, tmax, commSize)[commRank+1]


    TQ=0.4
    beta=1.0/T
    lattice,gens = makeLattice(dim, phdim)
    lattice.Qmax = sqrt(TQ/minimum(lattice.springConstants))

    lattice.Qmax=10.0



    m=MonteCarlo(lattice,beta,thermSweeps,sampleSweeps,dim,replicaExchangeRate=10)
    m.sampleRate=100
    run!(m,gens,dim, phdim)
    # e,e2=means(m.observables.energy)
    # print("Final Average energy: ", e, "\nFinal Average energy squared: ", e2, "\n")


    # c(e) = beta * beta * (e[2] - e[1] * e[1]) * length(m.lattice)
    # print("Magnetization vector: ", getMagnetization(m.lattice,dim), "\n")

    return (m, gens)
    # return mean(m.observables.energy, c)
end

# tstart=time()


# Tpoints=30
# Tvals = LinRange(2.0, 6.0, Tpoints)
# heat = zeros(Tpoints)

T=0.01
m,gens=runMC(T)
