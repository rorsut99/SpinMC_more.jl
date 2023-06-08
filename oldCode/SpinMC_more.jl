module SpinMC_more

include("UnitCell.jl")
export UnitCell, addInteraction!, setInteractionOnsite!, setField!, addBasisSite!
include("InteractionMatrix.jl")
include("Lattice.jl")
export Lattice, size, length, getSpin, setSpin!, getSitePosition, addGenerator!, addPhononInteraction!, addSpringConstant!

include("Observables.jl")
export Observables
include("Spin.jl")
export getEnergy, getMagnetization, getCorrelation

include("MonteCarlo.jl")
export MonteCarlo, run!

include("Helper.jl")
include("IO.jl")
export writeMonteCarlo, readMonteCarlo

using Reexport
@reexport using BinningAnalysis

end
