using BinningAnalysis

mutable struct EvolveObservables


spinStates::FullBinner{Matrix{Float64}}
phononPosition::FullBinner{Matrix{Float64}}
phononMomenta::FullBinner{Matrix{Float64}}

EvolveObservables()=new{}()
end

function initEvolveObservables()
    evsObs=EvolveObservables()

    evsObs.spinStates=FullBinner(Matrix{Float64})
    evsObs.phononPosition=FullBinner(Matrix{Float64})
    evsObs.phononMomenta=FullBinner(Matrix{Float64})

    return(evsObs)
end


function measureEvObservables!(evs)
    push!(evs.obs.spinStates, evs.lattice.expVals)
    push!(evs.obs.phononPosition, evs.lattice.phonons)
    push!(evs.obs.phononMomenta, evs.phononMomenta)
end