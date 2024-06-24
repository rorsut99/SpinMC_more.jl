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
    a1=(1.0,)

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

    uc = UnitCell(a1)


    addBasisSite!(uc,(0.0,),dim)

    int=Matrix(1.0I,3,3)
    addInteraction!(uc,gens,1,1,1*int,1,(1,))

    Lsize=(500,)
    lattice=Lattice(uc,Lsize,dim,phdim) 



    spring = Vector{Float64}()
    mat =Matrix{Float64}(undef,(4,0))
    addSpringConstant!(lattice, spring, phdim)
    addPhononInteraction!(lattice,1, gens, mat)
    return (lattice,gens)
end


function runMC(T)
    # define dimensions
    dim=2           # dimension of wavefunction (N)
    dim2=dim^2-1    # dimension of spin vector (N^2-1)
    phdim=0

    # set sweeps
    thermSweeps=3000
    sampleSweeps=7000


    TQ=0.4
    beta=1.0/T
    lattice,gens = makeLattice(dim, phdim)
    lattice.Qmax = 10

    m=MonteCarlo(lattice,beta,thermSweeps,sampleSweeps,dim)
    run!(m,gens,dim, phdim)
    # e,e2=means(m.observables.energy)
    # print("Final Average energy: ", e, "\nFinal Average energy squared: ", e2, "\n")


    # c(e) = beta * beta * (e[2] - e[1] * e[1]) * length(m.lattice)
    # print("Magnetization vector: ", getMagnetization(m.lattice,dim), "\n")

    return (m, gens)
    # return mean(m.observables.energy, c)
end


function runMD()

    m,gens = runMC(0.0001)
    T=1/m.beta
    print(T,"\n")
    dim=2
    phdim=0

    smp=initSMP(gens.dim,m.lattice,phdim)

    g4=1/160
    initPhMomentum!(smp,T,phdim,Vector{Float64}())
    setPhononMass!(smp,Vector{Float64}(),phdim)
    setPhononDamp!(smp,Vector{Float64}(),phdim)




    timeStep=0.1
    n = 5000
    timeSeries=Vector{Matrix{Float64}}()

    setDT!(smp,timeStep)

    spinEnergyi, phononEnergyi, coupledEnergyi, energyi = getEvEnergy(smp,gens,smp.lattice)
    measureEvObservables!(smp, spinEnergyi/length(m.lattice), phononEnergyi/length(m.lattice), coupledEnergyi/length(m.lattice), energyi/length(m.lattice))

    for i in 1:n
        finalState!(smp.lattice,gens)
        push!(timeSeries,smp.lattice.expVals)

        evolveSMP!(smp,smp.lattice,gens,3)

        spinEnergy, phononEnergy, coupledEnergy, energy = getEvEnergy(smp,gens,smp.lattice)
        measureEvObservables!(smp, spinEnergy/length(m.lattice), phononEnergy/length(m.lattice), coupledEnergy/length(m.lattice), energy/length(m.lattice))
    end

    # outfile="test.h5"
    # outfile*=string(n)
    # writeSMP(outfile,smp)
    return smp, timeSeries


end

smp, slices = runMD()

x=2
