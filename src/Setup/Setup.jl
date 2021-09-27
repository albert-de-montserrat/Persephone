struct ThermalParameters{T}
    κ::T
    α::T
    Cp::T
    dQdT::T
end

function thermal_parameters(; 
    κ = 1.0,
    α = 1e-6,
    Cp = 1.0,
    dQdT = 0.0,
)
    ThermalParameters(κ, α, Cp, dQdT)
end

function init_temperature(gr, IDs; type = :harmonic)
    
    θ = gr.θ
    r = gr.r
    
    Ttop, Tbot = 0.0, 1.0

    if type == :harmonic
        # Harmonic hermal perturbation
        y44 = @. 1.0/8.0*√(35.0/π)*cos(4.0*θ)
        δT = @. 0.2*y44*sin(π*(1-s))
        T = @. 1.22*s/(2.22-s) + δT
        s = @. (2.22-r)/(2.22-1.22)

    elseif type == :random
        # Linear temperature with random perturbation
        s = @. (2.22-r)/(2.22-1.22)
        T = s .* (1 .+ (rand(length(s)).-0.5).*0.01 )

    elseif type == :realistic
        transition = 2.22-0.2276
        T = @. (2.22-r)/(2.22-transition)
        idx = r .< transition
        T[idx] .= 1.0

    end
    
    fixT!(T, Ttop , Tbot, IDs)
    applybounds!(T, Tbot, Ttop)

    return T
end

function init_particle_temperature!(particle_fields, particle_info; type = :harmonic)
    
    θ = [particle_info[i].CPolar.x for i in eachindex(particle_info)]
    r = [particle_info[i].CPolar.z for i in eachindex(particle_info)]
    s = @. (2.22-r)/(2.22-1.22)
 
    if type == :harmonic
        # Harmonic hermal perturbation
        y44 = @. 1.0/8.0*√(35.0/π)*cos(4.0*θ)
        δT = @. 0.2*y44*sin(π*(1-s))
        particle_fields.T = @. 1.22*s/(2.22-s) + δT

    elseif type == :random
        # Linear temperature with random perturbation
        particle_fields.T = s .* (1 .+ (rand(length(s)).-0.5).*0.01 )

    end

end

function fixT!(T, Ttop, Tbot, IDs)
    @inbounds for i in axes(T,1)
        if IDs[i] == "inner"
            T[i] = Tbot
        elseif IDs[i] == "outter"
            T[i] = Ttop
        end
    end
end
