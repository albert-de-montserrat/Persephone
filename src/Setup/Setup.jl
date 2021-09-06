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
    s = @. (2.22-r)/(2.22-1.22)
    
    Ttop, Tbot = 0.0, 1.0
    
    # # Thermal perturbation
    # zsize = zmax-zmin
    # ival = @. z/zsize
    # y44 = [3/16 * sqrt(35/π)*cos(4GlobC[i].x) for i in axes(GlobC,1)]
    # δT = [0.1 * y44[i] * sin(π*(1.12-GlobC[i].z)) for i in axes(GlobC,1)]
    # T .+= δT
    # # T = @. 1 - (1.12*(ival/(2.22-ival)) + δT)

    if type == :harmonic
        # Harmonic hermal perturbation
        y44 = @. 1.0/8.0*√(35.0/π)*cos(4.0*θ)
        δT = @. 0.2*y44*sin(π*(1-s))
        T = @. 1.22*s/(2.22-s) + δT

    elseif type == :random
        # Linear temperature
        δT = s .* (1 .+ (rand(length(s)).-0.5).*0.01 )
        T = s .+ δT

    end
    
    fixT!(T, Ttop , Tbot, IDs)
    applybounds!(T, Tbot, Ttop)

    return T
end

function init_particle_temperature!(particle_fields, particle_info)
    
    x = [particle_info[i].CPolar.x for i in eachindex(particle_info)]
    z = [particle_info[i].CPolar.z for i in eachindex(particle_info)]
    s = @. (2.22-z)/(2.22-1.22)
 
    # Thermal perturbation - Jeff
    ival = s
    y44 = @. 1.0/8.0*√(35.0/π)*cos(4.0*x);
    δT = @. 0.2*y44*sin(π*(1-ival))
    particle_fields.T .= @. (1.22*(ival/(2.22-ival)) + δT)

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