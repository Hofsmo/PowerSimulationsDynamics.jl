
struct SmallSignalOutput
    reduced_jacobian::AbstractArray{Float64}
    eigenvalues::Vector{Complex{Float64}}
    eigenvectors::Matrix{Complex{Float64}}
    index::Dict
    stable::Bool
    operating_point::Vector{Float64}
    damping::Dict{String, Dict{Symbol, Float64}}
    participation_factors::Dict{String, Dict{Symbol, Array{Float64}}}
end

function _determine_stability(vals::Vector{Complex{Float64}})
    for real_eig in real(vals)
        real_eig > 0.0 && return false
    end
    return true
end

function _calculate_forwardiff_jacobian(
    sim::Simulation{ResidualModel},
    x_eval::Vector{Float64},
)
    var_count = get_variable_count(sim.inputs)
    dx0 = zeros(var_count) #Define a vector of zeros for the derivative
    sysf! = (out, x) -> system_implicit!(
        out,            #output of the function
        dx0,            #derivatives equal to zero
        x,              #states
        sim.inputs,     #Parameters
        0.0,            #time equals to zero.
    )
    out = zeros(var_count) #Define a vector of zeros for the output
    jacobian = ForwardDiff.jacobian(sysf!, out, x_eval)
    return jacobian
end

function _calculate_forwardiff_jacobian(
    sim::Simulation{MassMatrixModel},
    x_eval::Vector{Float64},
)
    var_count = get_variable_count(sim.inputs)
    sysf! = (dx, x) -> system_mass_matrix!(
        dx,            #derivatives equal to zero
        x,              #states
        sim.inputs,     #Parameters
        0.0,            #time equals to zero.
    )
    dx = zeros(var_count) #Define a vector of zeros for the output
    jacobian = ForwardDiff.jacobian(sysf!, dx, x_eval)
    return jacobian
end

"""
Finds the location of the differential states in the reduced Jacobian
"""
function _make_reduced_jacobian_index(global_index, diff_states)
    jac_index = Dict{String, Dict{Symbol, Int}}()
    for (device_name, device_index) in global_index
        jac_index[device_name] = Dict{Symbol, Int}()
        for (state, ix) in device_index
            state_is_differential = diff_states[ix]
            if state_is_differential
                jac_index[device_name][state] = sum(diff_states[1:ix])
            end
        end
    end
    return jac_index
end


function _get_eigenvalues(reduced_jacobian::AbstractArray{Float64}, multimachine::Bool)
    eigen_vals, R_eigen_vect = LinearAlgebra.eigen(Matrix(reduced_jacobian))
    if multimachine
        @warn(
            "No Infinite Bus found. Confirm stability directly checking eigenvalues.\nIf all eigenvalues are on the left-half plane and only one eigenvalue is zero, the system is small signal stable."
        )
        @debug(eigen_vals)
    end
    return eigen_vals, R_eigen_vect
end

function _get_damping(
    eigen_vals::Vector{Complex{Float64}},
    jac_index::Dict{String, Dict{Symbol, Int}},
)
    damping_results = Dict{String, Dict{Symbol, Float64}}()
    for (device_name, device_index) in jac_index
        damping_results[device_name] = Dict{Symbol, Float64}()
        for (state, ix) in device_index
            isnothing(ix) && continue
            eigen_val = eigen_vals[ix]
            damping_results[device_name][state] =
                -1 * real(eigen_val) / sqrt(real(eigen_val)^2 + imag(eigen_val)^2)
        end
    end
    return damping_results
end

function _get_participation_factors(
    R_eigen_vect::Matrix{Complex{Float64}},
    jac_index::Dict{String, Dict{Symbol, Int}},
)
    L_eigen_vect = inv(R_eigen_vect)
    participation_factors = Dict{String, Dict{Symbol, Array{Float64}}}()
    for (device_name, device_index) in jac_index
        participation_factors[device_name] = Dict{Symbol, Array{Float64}}()
        for (state, ix) in device_index
            den = sum(abs.(L_eigen_vect[:, ix]) .* abs.(R_eigen_vect[ix, :]))
            participation_factors[device_name][state] =
                abs.(L_eigen_vect[:, ix]) .* abs.(R_eigen_vect[ix, :]) ./ den
        end
    end
    return participation_factors
end

function _small_signal_analysis(
    jacobian::Matrix{Float64},
    x_eval::Vector{Float64},
    inputs::SimulationInputs,
    multimachine::Bool,
)
    mass_matrix = get_mass_matrix(inputs)
    diff_states = get_DAE_vector(inputs)
    global_index = make_global_state_map(inputs)
    jac_index = _make_reduced_jacobian_index(global_index, diff_states)
    reduced_jacobian = _reduce_jacobian(jacobian, diff_states, mass_matrix, global_index)
    eigen_vals, R_eigen_vect = _get_eigenvalues(reduced_jacobian, multimachine)
    damping = _get_damping(eigen_vals, jac_index)
    stable = _determine_stability(eigen_vals)
    participation_factors = _get_participation_factors(R_eigen_vect, jac_index)
    return SmallSignalOutput(
        reduced_jacobian,
        eigen_vals,
        R_eigen_vect,
        jac_index,
        stable,
        x_eval,
        damping,
        participation_factors,
    )
end

function _small_signal_analysis(
    ::Type{T},
    inputs::SimulationInputs,
    x_eval::Vector{Float64},
    multimachine = true,
) where {T <: SimulationModel}
    jacwrapper = get_jacobian(T, inputs, x_eval, 0)
    return _small_signal_analysis(jacwrapper.Jv, jacwrapper.x, inputs, multimachine)
end

function small_signal_analysis(sim::Simulation{T}; kwargs...) where {T <: SimulationModel}
    inputs = get_simulation_inputs(sim)
    x_eval = get(kwargs, :operating_point, get_initial_conditions(sim))
    return _small_signal_analysis(T, inputs, x_eval, sim.multimachine)
end

function small_signal_analysis(::Type{T}, system::PSY.System) where {T <: SimulationModel}
    simulation_system = deepcopy(system)
    inputs = SimulationInputs(T, simulation_system, ReferenceBus)
    x0_init = get_flat_start(inputs)
    set_operating_point!(x0_init, inputs, system)
    return _small_signal_analysis(T, inputs, x0_init)
end
