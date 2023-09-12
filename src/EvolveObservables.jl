using BinningAnalysis

mutable struct EvolveObservables


    spinStates::FullBinner{Matrix{Float64}}
    phononPosition::FullBinner{Matrix{Float64}}
    phononMomenta::FullBinner{Matrix{Float64}}
    totalEnergySeries::Vector{Float64}
    spinEnergySeries::Vector{Float64}
    coupledEnergySeries::Vector{Float64}
    phononEnergySeries::Vector{Float64}

    EvolveObservables()=new{}()
end

function initEvolveObservables()
    evsObs=EvolveObservables()

    evsObs.spinStates=FullBinner(Matrix{Float64})
    evsObs.phononPosition=FullBinner(Matrix{Float64})
    evsObs.phononMomenta=FullBinner(Matrix{Float64})
    evsObs.spinEnergySeries=Vector{Float64}()
    evsObs.phononEnergySeries=Vector{Float64}()
    evsObs.coupledEnergySeries=Vector{Float64}()
    evsObs.totalEnergySeries=Vector{Float64}()

    return(evsObs)
end

function getEvEnergy(evs,gens::Generators, lattice)
    spinEnergy = 0.0
    phEnergy = 0.0
    energy = 0.0
    coupleEnergy = 0.0
    d=size(gens.spinOperators[1])[1]
    Id=Matrix((1.0+0im)I,d,d)
    for site in 1:length(lattice)
        # get vector of exp values for site
        s0 = genExpVals(getSpin(evs.lattice, site),gens)
        p0 = getPhonon(lattice, site)
        # tempS0=copy(s0)
        
        # push!(tempS0,calcInnerProd(getSpin(lattice,site),Id,getSpin(lattice,site)))
    

        #two-spin interactions
        interactionSites = getInteractionSites(lattice, site)
        interactionMatrices = getInteractionMatrices(lattice, site)
        for i in 1:length(interactionSites)
            # get vector of exp values for interaction site
            s1 = genExpVals(getSpin(evs.lattice, interactionSites[i]),gens)
            # tempS1=copy(s1)
            # push!(tempS1,calcInnerProd(getSpin(lattice,interactionSites[i]),Id,getSpin(lattice,interactionSites[i])))
            if site > interactionSites[i]
                spinEnergy += exchangeEnergy(s0, interactionMatrices[i], s1)
            end
        end

        phEnergy += phononPotentialEnergy(lattice, p0)
        coupleEnergy += spinPhononCoupling(lattice, s0, p0)
        energy += spinPhononCoupling(lattice, s0, p0)
        phEnergy += sum((evs.phononMomenta[:,site] .^ 2) ./ (2 .* evs.phononMass))

        damping = evs.phononDamp
        phEnergy -= dot(damping.*p0, evs.phononMomenta[:,site])

        #onsite interaction
       # energy += exchangeEnergy(s0, getInteractionOnsite(lattice, site), s0)

        #field interaction
        spinEnergy += dot(s0, getInteractionField(evs.lattice, site))
    end

    energy += spinEnergy
    energy += phEnergy

    return spinEnergy, phEnergy, coupleEnergy, energy
end


function measureEvObservables!(evs, spinEnergy, phEnergy, coupleEnergy, energy)
    push!(evs.obs.spinStates, evs.lattice.expVals)
    push!(evs.obs.phononPosition, evs.lattice.phonons)
    push!(evs.obs.phononMomenta, evs.phononMomenta)
    push!(evs.obs.spinEnergySeries, spinEnergy)
    push!(evs.obs.phononEnergySeries, phEnergy)
    push!(evs.obs.coupledEnergySeries, coupleEnergy)
    push!(evs.obs.totalEnergySeries, energy)
end