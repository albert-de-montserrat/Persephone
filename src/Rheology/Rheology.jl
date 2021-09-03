abstract type Isotropic end
abstract type Anisotropic end

struct Isoviscous{T}
    val::Float64
end

struct TemperatureDependant{T}
    val::Vector{Float64}
end

Base.Val(x::Isoviscous{Isotropic}) = Val{Isotropic}()
Base.Val(x::Isoviscous{Anisotropic}) = Val{Anisotropic}()
Base.Val(x::TemperatureDependant{Isotropic}) = Val{Isotropic}()
Base.Val(x::TemperatureDependant{Anisotropic}) = Val{Anisotropic}()

function getviscosity(T, type; η=1)
    
    if type == "IsoviscousIsotropic"
        return Isoviscous{Isotropic}(float(η))

    elseif type == "IsoviscousAnisotropic"
        return Isoviscous{Anisotropic}(float(η))

    elseif type == "TemperatureDependantIsotropic"
        return TemperatureDependant{Isotropic}(@. exp(13.8156*(0.5-T)))
        # return TemperatureDependant{Isotropic}(@. exp(-log(1e5)*T))

    elseif type == "TemperatureDependantAnisotropic"
        return TemperatureDependant{Anisotropic}(@. exp(13.8156*(0.5-T)))
        # return TemperatureDependant{Anisotropic}(@. exp(1e3^(-T)))
        
    end

end

getviscosity!(η::Isoviscous, T) = nothing

function getviscosity!(η::TemperatureDependant, T) 
    @inbounds Threads.@threads for i in axes(η.val,1)
        η.val[i] = exp(13.8156*(0.5-T[i]))
        # η.val[i] = exp(-log(1e5)*T[i])
    end
end

# Temperature effect on density
# (α is thermal expansion coeff, ρ is defined at the nodes)
state_equation(α,T; ρ0 = 1) = @. ρ0 * (1-α*T);

state_equation!(ρ,α,T; ρ0 = 1) = ρ .= ones(length(T))