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
        scale=1
        sx=[0+0.0im 1.0+0im  0
            1.0+0im 0+0im 0
            0 0 0]/scale
        sy=[0 -1.0im 0
            1.0im 0 0
            0 0 0 ]/scale
        sz=[1.0+0im 0 0
            0 -1.0+0im 0
            0 0 0  ]/scale
        s4=[0+0im 0 1.0
            0 0 0
            1.0 0 0]/scale
        s5=[0 0 -1.0im
            0 0 0
            1.0im 0 0]/scale
        s6=[0+0.0im 0 0
            0 0 1.0
            0 1.0 0]/scale
        s7=[0 0 0
            0 0 -1.0im
            0 1.0im 0]/scale
        s8=[1.0+0im 0 0
            0 1.0 0
            0 0 -2.0]/sqrt(3)/scale
        s9 = Matrix(1.0I, 3, 3)
        # calculate norm
        normGens = sx^2 + sy^2 + sz^2 + s4^2 + s5^2 + s6^2 + s7^2 + s8^2 + s9^2
        norms=Real(normGens[1,1])
        
        
        generators = [sx, sy, sz, s4, s5, s6, s7, s8, s9]
    end
    return generators
end
function makeLattice(dim::Int,  phdim::Int, D::Float64)
    a1=(1.0,0,0)
    a2=(0,1.0,0)
    a3=(0,0,1.0)
    gens=initGen(dim)
    Sx=[0 0 1.0+0im
    0 0 1
    1 1 0]/sqrt(2)
    Sy=(-1im/sqrt(2))*[0 0 1.0+0im
    0 0 -1
    -1 1 0]
    Sz=[1.0+0im 0 0
    0 -1 0
    0 0 0]
    addSpinOperator!(gens,Sx)
    addSpinOperator!(gens,Sy)
    addSpinOperator!(gens,Sz)
    generators=makeGenerators(dim)
    for gen in generators
        addGenerator!(gens,gen)
    end
    setGenReps!(gens)
    uc = UnitCell(a1,a2,a3)
    addBasisSite!(uc,(0.0,0.0,0.0),dim)
    J1=1.0
    J2=0.1*J1
    #D=J1/(0.187*0.266)
    AFMint = Matrix(1.0I,3,3)
    addInteraction!(uc,gens,1,1,J1*AFMint,1,(1,0,0))
    addInteraction!(uc,gens,1,1,J1*AFMint,1,(0,1,0))
    addInteraction!(uc,gens,1,1,J2*AFMint,1,(0,0,1))
    setInteractionOnsite!(uc,1,D*Sz^2,dim,gens)
    
    Lsize=(14,14,6)
    lattice=Lattice(uc,Lsize,dim,phdim)
    spring = [0.0,0.0]
    # mat = [1.0; 0.0; 0.0 ;;]
    mat =[0.0 0.0
        0.0  0.0
        0.0  0.0]
    addSpringConstant!(lattice, spring, phdim)
    addPhononInteraction!(lattice,1, gens, mat)
    
    return (lattice,gens)
end
function runMC(T,D)
    dim=3           # dimension of wavefunction (N)
    dim2=dim^2-1    # dimension of spin vector (N^2-1)
    phdim=2
    MPI.Initialized() || MPI.Init()
    commSize = MPI.Comm_size(MPI.COMM_WORLD)
    commRank = MPI.Comm_rank(MPI.COMM_WORLD)
    # set sweeps
    thermSweeps=100000
    sampleSweeps=500000
    tmin=0.1
    tmax=1.0
    T=LinRange(tmin, tmax, commSize)[commRank+1]
    beta=1.0/T
    lattice,gens = makeLattice(dim, phdim,1/D)
    TQ=0.4
    lattice.Qmax = sqrt(TQ/minimum(lattice.springConstants))
    outf=string("BaFeSiO--Dratio=",D,".h5")
    lattice.Qmax=10.0
    m=MonteCarlo(lattice,beta,thermSweeps,sampleSweeps,dim,replicaExchangeRate=10)
    run!(m,gens,dim, phdim,outfile=outf)
    
    return(m,gens)
end
D=parse(Float64,ARGS[1])
tick=time()
m,gens=runMC(0.01,D)
tock=time()
print(tock-tick)