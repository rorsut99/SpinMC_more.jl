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
        s9 = Matrix(1.0I, 2, 2)

        # calculate norm
        norm = sx^2 + sy^2 + sz^2 + s9^2

        generators = [sx, sy, sz, s9]

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
        s9 = Matrix(1.0I, 3, 3)

        # calculate norm
        norm = sx^2 + sy^2 + sz^2 + s4^2 + s5^2 + s6^2 + s7^2 + s8^2 + s9^2
        print(norm)
     
        generators = [sx, sy, sz, s4, s5, s6, s7, s8, s9]
    end

    return generators
end

function makeLattice(dim::Int, dim2::Int, phdim::Int)

    # define cubic lattice with Heisenberg interaction
    a1=(sqrt(3)/2,1.0/2)
    a2=(sqrt(3)/2,-1.0/2)

    gens=initGen(dim)

    Sx=[0+0.0im 1.0+0im 
    1.0+0im 0+0im]
    Sy=[0 -1.0im
    1.0im 0]
    Sz=[1.0+0im 0
    0 -1.0+0im]

    # Sx=[0 0 0.0+0im
    # 0 0 -1im
    # 0 1im 0]

    # Sy=[0 0 0.0+1.0im
    # 0 0 0
    # -1im 0 0]

    # Sz=[0.0+0im -1im 0
    # 1im 0 0
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
    FMint = Matrix(-1.0I,3,3)      # Heisenberg interaction
    AFMint = Matrix(1.0I,3,3)      # Heisenberg interaction
    Zero = Matrix(0.0I,dim,dim)
    addBasisSite!(uc,(0.0,0.0),dim)
    # nearest neighbour interaction
    # added parameter to take in the order of the term in the hamiltonian
    # bilinear interaction
    J1 = 0.0
    addInteraction!(uc,gens,1,1,J1*AFMint,1,(1,0))
    addInteraction!(uc,gens,1,1,J1*AFMint,1,(0,1))
    addInteraction!(uc,gens,1,1,J1*AFMint,1,(1,-1))

    B=[0.0,0,0]
    setField!(uc,gens,1,B)

    # # biquadratic interaction
    J2 = 0.0
    addInteraction!(uc,gens,1,1,J2*FMint,2,(1,0))
    addInteraction!(uc,gens,1,1,J2*FMint,2,(0,1))
    addInteraction!(uc,gens,1,1,J2*FMint,2,(1,-1))
 



    

    Lsize=(5,5)       # size of lattice
    lattice=Lattice(uc,Lsize,dim,phdim)

    

    spring = [1.0,1.0]
    # mat = [1.0; 0.0; 0.0 ;;]
    mat =[0.0 0.0
        0.0  0.0
        0.0  0.0]

    addSpringConstant!(lattice, spring, phdim)
    addPhononInteraction!(lattice,1, gens, mat)
    return (lattice,gens)
end

function runMC(T)
    # define dimensions
    dim=2          # dimension of wavefunction (N)
    dim2=dim^2-1    # dimension of spin vector (N^2-1)

    phdim=2


    # set sweeps
    thermSweeps=200
    sampleSweeps=200



    beta=1.0/T


    lattice,gens = makeLattice(dim, dim2, phdim)
    lattice.Qmax = 0.05

    # run Monte Carlo sweeps
    m=MonteCarlo(lattice,beta,thermSweeps,sampleSweeps,dim,replicaExchangeRate=10)
    run!(m,gens,dim, phdim)
    e,e2=means(m.observables.energy)


    return(m,gens)

end

function runMD(m,gens,T)
    smp=initSMP(gens.dim,m.lattice,2)

    initPhMomentum!(smp,T,2,[0.0,0.0])
    setPhononMass!(smp,[1.0,1.0],2)
    setPhononDamp!(smp,[0.1,0.1],2)

    function cosineDrive(t)
        A=0.01
        w=1.0
        phi=0.0
        x=A*cos(w*t+phi)
        return(x)
    end

    function noDrive(t)
        x=0
        return(x)
    end

    funcs=[cosineDrive,noDrive]
    addPhononDrive!(smp,funcs,2)

    timeStep=0.01
    n = 25000

    phx1=zeros(n)
    phz1=zeros(n)

    setDT!(smp,timeStep)

    for i in 1:n
        evolveSMP!(smp,smp.lattice,gens,2)
        phx1[i]=smp.lattice.phonons[1,1]
        updateTimeSpan!(smp,smp.dt)
    end
    return(phx1)
end


T=2.0
m,gens=runMC(T)



phx1=runMD(m,gens,T)