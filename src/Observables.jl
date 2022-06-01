using BinningAnalysis

mutable struct Observables
    energy::ErrorPropagator{Float64,32}
    magnetization::LogBinner{Float64,32,BinningAnalysis.Variance{Float64}}
    #
    mx::LogBinner{Float64,32,BinningAnalysis.Variance{Float64}}
    my::LogBinner{Float64,32,BinningAnalysis.Variance{Float64}}
    mz::LogBinner{Float64,32,BinningAnalysis.Variance{Float64}}
    
    chi_xx::LogBinner{Float64,32,BinningAnalysis.Variance{Float64}}
    chi_xz::LogBinner{Float64,32,BinningAnalysis.Variance{Float64}}
    chi_zx::LogBinner{Float64,32,BinningAnalysis.Variance{Float64}}
    chi_zz::LogBinner{Float64,32,BinningAnalysis.Variance{Float64}}
    #
    magnetizationVector::LogBinner{Vector{Float64},32,BinningAnalysis.Variance{Vector{Float64}}}
    correlation::LogBinner{Vector{Float64},32,BinningAnalysis.Variance{Vector{Float64}}}
end

function Observables(lattice::T) where T<:Lattice
    return Observables(ErrorPropagator(Float64), LogBinner(Float64), 
        LogBinner(Float64), LogBinner(Float64), LogBinner(Float64), #M components
        LogBinner(Float64), LogBinner(Float64), LogBinner(Float64),LogBinner(Float64), #Chi tensor
        LogBinner(zeros(Float64,3)), LogBinner(zeros(Float64,lattice.length))) 
end

function performMeasurements!(observables::Observables, lattice::T, energy::Float64) where T<:Lattice
    #measure energy and energy^2
    push!(observables.energy, energy / length(lattice), energy * energy / (length(lattice) * length(lattice)))

    #measure magnetization
    m = getMagnetization(lattice)
    push!(observables.magnetization, norm(m))
    #
    push!(observables.mx, m[1])
    push!(observables.my, m[2])
    push!(observables.mz, m[3])
    push!(observables.chi_xx, getSusceptibility(1,1,lattice))
    push!(observables.chi_xz, getSusceptibility(1,3,lattice))
    push!(observables.chi_zx, getSusceptibility(3,1,lattice))
    push!(observables.chi_zz, getSusceptibility(3,3,lattice))
    #
    push!(observables.magnetizationVector, m)

    #measure spin correlations
    push!(observables.correlation, getCorrelation(lattice))
end
