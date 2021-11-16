abstract type AbstractViscosity end
abstract type Isotropic end
abstract type Anisotropic end

struct Isoviscous{T} <: AbstractViscosity  
    val::Float64
end

struct TemperatureDependant{T} <: AbstractViscosity 
    val::Vector{Float64}
end

struct IsoviscousPlastic{T} <: AbstractViscosity  
    node::Float64
    ip::Matrix{Float64}
    τ_VonMises::Float64

    function IsoviscousPlastic(T, val, τ_VonMises, nel, nip)
        new{T}(val, zeros(nel, nip), τ_VonMises)
    end
end

struct TemperatureDependantPlastic{T} <: AbstractViscosity 
    node::Vector{Float64}
    ip::Matrix{Float64}
    τ_VonMises::Float64

    function TemperatureDependantPlastic(T, val, τ_VonMises, nel, nip)
        new{T}(val, zeros(nel, nip), τ_VonMises)
    end
end

struct Mallard2016{T} <: AbstractViscosity 
    node::Vector{Float64}
    ip::Matrix{Float64}
    τ_VonMises::Float64

    function TemperatureDependantPlastic(T, val, τ_VonMises, nel, nip)
        new{T}(val, zeros(nel, nip), τ_VonMises)
    end
end

Base.Val(x::Isoviscous{Isotropic})             = Val{Isotropic}()
Base.Val(x::Isoviscous{Anisotropic})           = Val{Anisotropic}()
Base.Val(x::TemperatureDependant{Isotropic})   = Val{Isotropic}()
Base.Val(x::TemperatureDependant{Anisotropic}) = Val{Anisotropic}()
Base.Val(x::TemperatureDependantPlastic{Isotropic}) = Val{Isotropic}()

function getviscosity(T, type, nel; η = 1, τ_VonMises = 5e3, nip = 7)
    
    if type == :IsoviscousIsotropic
        return Isoviscous{Isotropic}(float(η))

    elseif type == :IsoviscousAnisotropic
        return Isoviscous{Anisotropic}(float(η))

    elseif type == :TemperatureDependantIsotropic
        return TemperatureDependant{Isotropic}(@. exp(13.8156*(0.5-T)))

    elseif type == :TemperatureDependantIsotropicPlastic
        # return TemperatureDependantPlastic(Isotropic, @.(exp(13.8156*(0.5-T))), τ_VonMises, nel, nip)
        return TemperatureDependantPlastic(Isotropic, @.(exp(23.03/(1+T) -23.03/2)), τ_VonMises, nel, nip)

    elseif type == :VanHeckIsotropic
        return TemperatureDependant{Isotropic}(@. exp(23.03/(1+T) -23.03/2))

    elseif type == :TemperatureDependantAnisotropic
        return TemperatureDependant{Anisotropic}(@. exp(13.8156*(0.5-T)))

    elseif type == :TemperatureDependantAnisotropicPlastic
        # return TemperatureDependant(Anisotropic, @.(exp(13.8156*(0.5-T))), τ_VonMises, nel, nip)
        return TemperatureDependant(Anisotropic, @.(exp(23.03/(1+T) -23.03/2)), τ_VonMises, nel, nip)
    
    elseif type == :VanHeckAnisotropic
        return TemperatureDependant{Anisotropic}(@. exp(23.03/(1+T) -23.03/2))
        
    end

end

getviscosity!(η::Isoviscous, T) = nothing

function getviscosity!(η::TemperatureDependant, T) 
    Threads.@threads for i in eachindex(η.val)
        # update viscosity
        @inbounds η.val[i] = exp(13.8156*(0.5-T[i]))
    end
end

function getviscosity!(η::TemperatureDependantPlastic, T, r)
    @inbounds for i in eachindex(η.node)
        depth = r[i] - 1.22
        # viscosity correction from Richards et al. [2001] and Tackley [2000b] 
        m = T[i] < 0.6 + 2*(1-depth) ? 1 : 0.1 
        # update viscosity
        η.node[i] = exp(23.03/(1+T[i]) -23.03/2) * m
    end
end

function getviscosity!(η::Mallard2016, T, r)

    # Parameters from Mallard et al 2016 - Nature
    a, B, d0, dstep = 1e6, 30, 0.276, 0.02

    # η(z,T) from Mallard et al 2016 - Nature
    @inbounds for i in eachindex(η)
        depth = 2.22 - r[i]
        # viscosity correction 
        m = T[i] < 0.6 + 7.5*depth ? 1 : 0.1 
        # depth dependent component
        ηz = a*exp(log(B)*(1-0.5*(1-tanh((d0-depth)/dstep))))
        # update viscosity
        η[i] = ηz*exp(0.064 - 30/(T[i]+1))*m
    end
end

# Temperature effect on density
# (α is thermal expansion coeff, ρ is defined at the nodes)
state_equation(α, T; ρ0 = 1) = @. ρ0 * (1-α*T);

state_equation!(ρ,α,T; ρ0 = 1) = ρ .= ones(length(T))