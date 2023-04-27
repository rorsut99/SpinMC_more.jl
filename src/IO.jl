using HDF5
using Serialization

function array(tuple::NTuple{N,T}) where {N,T<:Number}
    return [ x for x in tuple]
end

function writeMonteCarlo(filename::String, mc::MonteCarlo{Lattice{D,N,dim}}) where {D,N}
    h5open(filename, "w") do f
        #write binary checkpoint
        data = IOBuffer()
        serialize(data, mc)
        f["checkpoint"] = take!(data)

        #write human readable results and parameters
        f["mc/beta"] = mc.beta
        f["mc/thermalizationSweeps"] = mc.thermalizationSweeps
        f["mc/measurementSweeps"] = mc.measurementSweeps
        f["mc/measurementRate"] = mc.measurementRate
        f["mc/reportInterval"] = mc.reportInterval
        f["mc/checkpointInterval"] = mc.checkpointInterval
        f["mc/seed"] = mc.seed
        f["mc/sweep"] = mc.sweep

        f["mc/lattice/L"] = array(mc.lattice.size)
        for i in 1:D
            f["mc/lattice/unitcell/primitive/"*string(i)] = array(mc.lattice.unitcell.primitive[i])
        end
        for i in 1:length(mc.lattice.unitcell.basis)
            f["mc/lattice/unitcell/basis/"*string(i)] = array(mc.lattice.unitcell.basis[i])
            f["mc/lattice/unitcell/interactionsOnsite/"*string(i)] = mc.lattice.unitcell.interactionsOnsite[i]
            f["mc/lattice/unitcell/interactionsField/"*string(i)] = mc.lattice.unitcell.interactionsField[i]
        end
        for i in 1:length(mc.lattice.unitcell.interactions)
            f["mc/lattice/unitcell/interactions/"*string(i)*"/b1"] = mc.lattice.unitcell.interactions[i][1]
            f["mc/lattice/unitcell/interactions/"*string(i)*"/b2"] = mc.lattice.unitcell.interactions[i][2]
            f["mc/lattice/unitcell/interactions/"*string(i)*"/offset"] = array(mc.lattice.unitcell.interactions[i][3])
            f["mc/lattice/unitcell/interactions/"*string(i)*"/M"] = mc.lattice.unitcell.interactions[i][4]
        end
        for i in 1:length(mc.lattice)
            f["mc/lattice/sitePositions/"*string(i)] = array(mc.lattice.sitePositions[i])
        end

        f["mc/observables/expVals"] = array(mc.lattice.expVals)

        f["mc/observables/energyDensity/mean"] = means(mc.observables.energy)[1]
        f["mc/observables/energyDensity/error"] = std_errors(mc.observables.energy)[1]
        f["mc/observables/magnetization/mean"] = mean(mc.observables.magnetization)
        f["mc/observables/magnetization/error"] = std_error(mc.observables.magnetization)
        f["mc/observables/magnetizationVector/mean"] = mean(mc.observables.magnetizationVector)
        f["mc/observables/magnetizationVector/error"] = std_error(mc.observables.magnetizationVector)
        #
        f["mc/observables/mx/mean"] = mean(mc.observables.mx)
        f["mc/observables/mx/error"] = std_error(mc.observables.mx)
        f["mc/observables/my/mean"] = mean(mc.observables.my)
        f["mc/observables/my/error"] = std_error(mc.observables.my)
        f["mc/observables/mz/mean"] = mean(mc.observables.mz)
        f["mc/observables/mz/error"] = std_error(mc.observables.mz)
        # f["mc/observables/chi_xx/mean"] = mean(mc.observables.chi_xx)
        # f["mc/observables/chi_xx/error"] = std_error(mc.observables.chi_xx)
        # f["mc/observables/chi_xz/mean"] = mean(mc.observables.chi_xz)
        # f["mc/observables/chi_xz/error"] = std_error(mc.observables.chi_xz)
        # f["mc/observables/chi_zx/mean"] = mean(mc.observables.chi_zx)
        # f["mc/observables/chi_zx/error"] = std_error(mc.observables.chi_zx)
        # f["mc/observables/chi_zz/mean"] = mean(mc.observables.chi_zz)
        # f["mc/observables/chi_zz/error"] = std_error(mc.observables.chi_zz)
        f["mc/observables/chitens/mean"] = mean(mc.observables.chitens)
        f["mc/observables/chitens/error"] = std_error(mc.observables.chitens)
        
        f["mc/observables/txlist/mean"] = mean(mc.observables.txlist)
        f["mc/observables/txlist/error"] = std_error(mc.observables.txlist)
        f["mc/observables/tylist/mean"] = mean(mc.observables.tylist)
        f["mc/observables/tylist/error"] = std_error(mc.observables.tylist)
        f["mc/observables/tzlist/mean"] = mean(mc.observables.tzlist)
        f["mc/observables/tzlist/error"] = std_error(mc.observables.tzlist)
        
        #
        f["mc/observables/correlation/mean"] = mean(mc.observables.correlation)
        f["mc/observables/correlation/error"] = std_error(mc.observables.correlation)

        c(e) = mc.beta * mc.beta * (e[2] - e[1] * e[1]) * length(mc.lattice)
        ∇c(e) = [-2.0 * mc.beta * mc.beta * e[1] * length(mc.lattice), mc.beta * mc.beta * length(mc.lattice)]
        heat = mean(mc.observables.energy, c)
        dheat = sqrt(abs(var(mc.observables.energy, ∇c, BinningAnalysis._reliable_level(mc.observables.energy))) / mc.observables.energy.count[BinningAnalysis._reliable_level(mc.observables.energy)])
        f["mc/observables/specificHeat/mean"] = heat
        f["mc/observables/specificHeat/error"] = dheat
    end
end

function readMonteCarlo(filename::String)
    h5open(filename, "r") do f
        data = IOBuffer(read(f["checkpoint"]))
        return deserialize(data)
    end
end
