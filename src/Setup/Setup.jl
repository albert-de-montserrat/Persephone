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

function init_temperature(gr, IDs)
    
    θ = gr.θ
    r = gr.r
    s = @. (2.22-r)/(2.22-1.22)
    # rmin, rmax = extrema(r)
    
    Ttop, Tbot = 0.0, 1.0
    
    # # Linear temperature
    # m = Ttop - Tbot
    # b = Ttop - 2.22m
    # T = @. m*z+b 
    
    # # Thermal perturbation
    # zsize = zmax-zmin
    # ival = @. z/zsize
    # y44 = [3/16 * sqrt(35/π)*cos(4GlobC[i].x) for i in axes(GlobC,1)]
    # δT = [0.1 * y44[i] * sin(π*(1.12-GlobC[i].z)) for i in axes(GlobC,1)]
    # T .+= δT
    # # T = @. 1 - (1.12*(ival/(2.22-ival)) + δT)

    # Thermal perturbation - Jeff
    y44 = @. 1.0/8.0*√(35.0/π)*cos(4.0*θ)
    δT = @. 0.2*y44*sin(π*(1-s))
    # T = @. (round(1.22*(s/(2.22-s)), digits=3) + δT)
    T = @. 1.22*s/(2.22-s) + δT

    # A, N = 0.1, 4
    # δT = @. A*s*(1-s)*cos(N*x)
    # T = @. s + δT 

    # # Random noise
    # max_percent = 0.02
    # δT = max_percent*T.*(rand(length(T)).-0.5)

    # add perturbation
    # T .+= δT
    
    fixT!(T, Ttop , Tbot, IDs)
    # # T = max.(T,Ttop)
    # # T = min.(T,Tbot)
    applybounds!(T, Tbot, Ttop)

    return T
end

function init_particle_temperature!(particle_fields, particle_info)
    
    x = [particle_info[i].CPolar.x for i in eachindex(particle_info)]
    z = [particle_info[i].CPolar.z for i in eachindex(particle_info)]
    s = @. (2.22-z)/(2.22-1.22)
    # zmin, zmax = extrema(z)
    
    # Ttop, Tbot = 0.0, 1.0
    
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

function domain_ids(GlobC)

    top = "outter"
    bot = "inner"
    inner = "inside"

    nnod = length(GlobC)
    IDs = Vector{String}(undef,nnod)

    r = round.([GlobC[i].z for i in axes(GlobC,1)],digits=3)

    rmax,rmin = maximum(r), minimum(r)
    
    Threads.@threads for i in axes(GlobC,1)
        if r[i] == rmax
            IDs[i] = top
        elseif r[i] == rmin
            IDs[i] = bot
        else
            IDs[i] = inner
        end
    end

    return IDs
end
