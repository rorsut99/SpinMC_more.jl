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


    Tpoints=30
    Tvals=zeros(Tpoints)
    dheat=zeros(Tpoints)
    My=zeros(Tpoints)
    Mx=zeros(Tpoints)
    Mz=zeros(Tpoints)
    dM=zeros(Tpoints)

    heat = zeros(Tpoints)
    energy = zeros(Tpoints)
    for i in 0:29
        stem="strongQuadDat/pseudoSpin--strongQuad--K1=1.h5.."
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
        My[i+1]=mean(m.observables.my)[1]
        Mx[i+1]=mean(m.observables.mx)[1]
        Mz[i+1]=mean(m.observables.mz)[1]
        dM[i+1]= std_error(m.observables.my)


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

    # plot Y-Magnetization
    # plot!(Tvals,abs.(My),yerr=dM,label="My")
    # xlabel!("T (J)")
    # ylabel!("My")
    # return(Tvals,My,Mx,Mz)
end 

plotDat()










