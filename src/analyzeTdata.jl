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

using HDF5
using Serialization
using JLD2
using Plots
using FFTW
using Peaks


function pltTdat()
    Tmin=1.0
    Tmax=8.0
    dt=0.1
    nt=71

    Ts=zeros(nt)
    for i in 1:nt
        Ts[i]=Tmin+(i-1)*dt
    end

    maxPHZ2=zeros(nt)


    for i in 1:nt
        T=round(Ts[i],digits=2)
        filename=string("TdataMultipolar/averagedPseudoDatT=",T,".h5")
        f = jldopen(filename, "r")
        PHZ2=f["totalPHZ2"]

        aPHZ=sqrt.(PHZ2)

        F = fftshift(fft(aPHZ))
        
        maxPHZ2[i] = maximum(abs.(F)[4800:end])


        # temp=maximum(PHZ2)
        # maxSz2[i]=temp
    end
    return(Ts,maxPHZ2)
end 
        

T,PHZ=pltTdat()

freqs = fftshift(fftfreq(5000,100))