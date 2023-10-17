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
using Plots
using HDF5
using Serialization
using JLD2


function plotDat()
    peakPos=0
    peak=0


    nFiles=40
    n=5001
    totalSX = zeros(n)
    totalSY = zeros(n)
    totalSZ = zeros(n)
    totalPHX = zeros(n)
    totalPHZ = zeros(n)
    totalPMX = zeros(n)
    totalPMZ = zeros(n)
    totalPHZ2 = zeros(n)
    
    

    for i in 0:nFiles-1
        stem="DynamicsDat/pseudoDynamics--dimlessVars--T=8.h5."
        app=string(i)
        filename=stem*app
        smp = readSMP(filename)



        totalSX+=smp.obs.avgSX
        totalSY+=smp.obs.avgSY
        totalSZ+=smp.obs.avgSZ
        


        for j in 1:5001

            totalPHX[j]+=smp.obs.avgPhononQ[j][1]
            totalPHZ[j]+=smp.obs.avgPhononQ[j][2]
            totalPHZ2[j]+=smp.obs.avgPhononQ[j][2]^2
        end
    

    end

    totalSX/=nFiles
    totalSY/=nFiles
    totalSZ/=nFiles
    totalPHX/=nFiles
    totalPHZ/=nFiles
    totalPHZ2/=nFiles
    return(totalSX,totalSY,totalSZ,totalPHX,totalPHZ,totalPHZ2)
end 

Sx,Sy,Sz,PHX,PHZ,PHZ2=plotDat()