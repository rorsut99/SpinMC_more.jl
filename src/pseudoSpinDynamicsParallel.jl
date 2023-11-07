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
        
        # push generators to lattice struct
        for gen in generators
            addGenerator!(gens,gen)
        end
    end

    setGenReps!(gens)
    return(gens)
end


function runMD(filename,outfile)

    MPI.Initialized() || MPI.Init()
    commSize = MPI.Comm_size(MPI.COMM_WORLD)
    commRank = MPI.Comm_rank(MPI.COMM_WORLD)

    filename*=string(commRank)
    outfile*=string(commRank)

    m=readMonteCarlo(filename)
    T=1/m.beta
    print(T,"\n")
    dim=2

    gens=makeGenerators(dim)
    smp=initSMP(gens.dim,m.lattice,2)


    g4=1/160
    initPhMomentum!(smp,T,2,[0.5,0.0])
    setPhononMass!(smp,[g4,g4],2)
    setPhononDamp!(smp,[0.1,0.0],2)




    timeStep=0.01
    n = 5000

    phx1=zeros(n)
    phz1=zeros(n)

    setDT!(smp,timeStep)



    spinEnergyi, phononEnergyi, coupledEnergyi, energyi = getEvEnergy(smp,gens,smp.lattice)
    measureEvObservables!(smp, spinEnergyi/length(m.lattice), phononEnergyi/length(m.lattice), coupledEnergyi/length(m.lattice), energyi/length(m.lattice))

    avgPHX=zeros(n)
    avgPHZ=zeros(n)

    avgPMX=zeros(n)
    avgPMZ=zeros(n)

    avgSX=zeros(n)
    avgSY=zeros(n)
    avgSZ=zeros(n)

    for i in 1:n
        finalState!(smp.lattice,gens)
        # avgPHX[i]=mean(smp.lattice.phonons[1,:])
        # avgPHZ[i]=mean(smp.lattice.phonons[2,:])

        # avgPMX[i]=mean(smp.phononMomenta[1,:])
        # avgPMZ[i]=mean(smp.phononMomenta[2,:])

        # avgSX[i]=mean(smp.lattice.expVals[1,:])
        # avgSY[i]=mean(smp.lattice.expVals[2,:])
        # avgSZ[i]=mean(smp.lattice.expVals[3,:])

        # phx1[i]=smp.lattice.phonons[1,1]
        # phz1[i]=smp.lattice.phonons[2,1]




        evolveSMP!(smp,smp.lattice,gens,3)

        spinEnergy, phononEnergy, coupledEnergy, energy = getEvEnergy(smp,gens,smp.lattice)
        measureEvObservables!(smp, spinEnergy/length(m.lattice), phononEnergy/length(m.lattice), coupledEnergy/length(m.lattice), energy/length(m.lattice))
    end

    writeSMP(outfile,smp)
    # return smp, avgSX, avgSY, avgSZ, avgPHX, avgPHZ, avgPMX, avgPMZ


end

T=ARGS[1]


file=string("pseudoSpinMCdatT=",T,"/pseudoSpin--dimlessVars--T=",T,".h5.")
outfile=string("pseudoDynamics--dimlessVars--T=",T,".h5.")
runMD(file,outfile)

# totalSX = zeros(5000)
# totalSY = zeros(5000)
# totalSZ = zeros(5000)
# totalPHX = zeros(5000)
# totalPHZ = zeros(5000)
# totalPMX = zeros(5000)
# totalPMZ = zeros(5000)


# for i in 1:40
#     file = string("data/pseudoSpin-T=4.h5.", string(i-1))
#     smp, SX, SY, SZ, PHX, PHZ, PMX, PMZ = runMD(file)

#     global totalSX += SX
#     global totalSY += SY
#     global totalSZ += SZ
#     global totalPHX += PHX
#     global totalPHZ += PHZ
#     global totalPMX += PMX
#     global totalPMZ += PMZ

# end

# avgSX = totalSX./40
# avgSY = totalSY./40
# avgSZ = totalSZ./40
# avgPHX = totalPHX./40
# avgPHZ = totalPHZ./40
# avgPMX = totalPMX./40
# avgPMZ = totalPMZ./40
