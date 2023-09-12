using Plots
using HDF5
using Serialization
include("UnitCell.jl")
include("Generators.jl")
include("InteractionMatrix.jl")
include("Lattice.jl")
include("Spin.jl")
include("Observables.jl")
include("MonteCarlo.jl")
include("Helper.jl")
include("IO.jl")
include("Phonon.jl")


function plotDat()
    peakPos=0
    peak=0


    Tpoints=80
    Tvals=zeros(Tpoints)
    dheat=zeros(Tpoints)
    
    heat = zeros(Tpoints)
    energy = zeros(Tpoints)
    for i in 0:79
        stem="data/coupling.h5."
        app=string(i)
        filename=stem*app
        m = readMonteCarlo(filename)
        beta=m.beta
        c(e) = beta * beta * (e[2] - e[1] * e[1]) * length(m.lattice)
        ∇c(e) = [-2.0 * beta * beta * e[1] * length(m.lattice), beta * beta * length(m.lattice)]
        # if i==0
        #     energySeries=m.energySeries
        #     display(plot(m.energySeries))
        #     print(1.0/beta)
        # end


        energy[i+1],temp = means(m.observables.energy)

        heat[i+1] = mean(m.observables.energy, c)
        dheat[i+1] = std_error(m.observables.energy, ∇c)
        Tvals[i+1]=1.0/beta



        if (Tvals[i+1]>0.4)
            if (heat[i+1]> peak)
                peakPos=Tvals[i+1]
                peak=heat[i+1]
            end
        end

    end

    print(peakPos)

    # plot energy vs sweeps
    plot!(Tvals,heat,yerr=dheat,label="Coupling")
    print(heat)
    xlabel!("T (J)")
    ylabel!("C")
end 

plotDat()










