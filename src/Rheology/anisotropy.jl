mutable struct StiffnessTensor{T}
    η11::Matrix{T}
    η33::Matrix{T}
    η55::Matrix{T}
    η13::Matrix{T}
    η15::Matrix{T}
    η35::Matrix{T}
end

struct Parameterisation{T}
    ζ::T
    ξ::T
    χ::T
end

"""
    Initialise 2D isotropic viscous tensor
"""
𝓒init(nel,nip) = StiffnessTensor(fill(4/3,nel,nip), 
                                 fill(4/3,nel,nip), 
                                 fill(1.0,nel,nip), 
                                 fill(-2/3,nel,nip), 
                                 fill(0.0,nel,nip), 
                                 fill(0.0,nel,nip))

"""
    𝓒 = Tensor: [η11 η22 η33 η44 η55 η66 η12 η13 η23] 
    a = Semi-axes: [a₁ a₂ a₃]
    ϕ = Volume fraction
    w = viscosity cut-off
"""
struct DEM 
    𝓒::Array{Float64,2}  
    a::Array{Float64,2} 
    ϕ::Vector{Float64} 
    w::Float64 
    a1a2_blk::Vector{Float64}
    a2a3_blk::Vector{Float64}
    permutation_blk::Vector{Int64} 
    sblk::Int64
    nblk::Int64
end

"""
    DEMloader(fname::String) 
        Load DEM hdf5 file
        fname -> file name, including path and extension
"""
function DEMloader(fname::String)
    fid = h5open(fname,"r")
    tensor = read(fid,"C")
    axes = read(fid,"ax")
    volume = read(fid,"vol")
    return tensor,axes,volume
end

"""
    weakening(Δη::Float64, ϕ::Number) 
        Compute viscosity cutoff as function of volume fraction
"""
function weakening(Δη::Float64, ϕ0::Real)
    ϕ = ifelse(ϕ0 < 1, ϕ0, ϕ0/100)

    if ϕ == 0.10
        ζ =  1.111
        ξ =  1.109
        χ = -1.597
        θ =  0.3754
        ψ =  0.699
        λ =  0.6834
        
    elseif ϕ == 0.20 
        ζ =  43.646559
        ξ = -42.907958
        χ =  0.1
        θ =  0.261095
        ψ =  0.984129
        λ =  1.0
        
    elseif ϕ == 0.30 
        ζ =  40.628095
        ξ = -39.823767
        χ =  0.1
        θ =  0.196188
        ψ =  0.985677
        λ =  1.0
    end

    return ζ*Δη^ψ + ξ*Δη^λ + χ*Δη + θ
end

"""
    getDEM(fname::String)
        Initialise DEM structure reading .h5 from fname
"""
function getDEM(fname::String)
    tensor,axis,volume = DEMloader(fname)
    # fixtensor!(tensor)

    # tmp = deepcopy(tensor)
    # tensor[:,1] = tmp[:,1] - tmp[:,7] - tmp[:,8]
    # tensor[:,3] = tmp[:,3] - tmp[:,9] - tmp[:,8]
    # tensor[:,7] .= 0.0
    # tensor[:,8] .= 0.0
    # tensor[:,9] .= 0.0

    w = weakening(1e-3, 0.20)

    # -- Get and process axes from the DEM
    ax1   = view(axis,:,1)
    ax2   = 1.0
    ax3   = view(axis,:,3)
    a1a2  = @. log10(ax1/ax2)
    a2a3  = @. log10(ax2/ax3)
    a2a3_blk, permutation_blk, sblk, nblk = sort_axes(a2a3,a1a2)
    idx = 1:sblk:length(a2a3)   
    # idx = 1:sblk
    a1a2_blk = view(a1a2,idx)

    DEM(
        tensor,
        axis, 
        dropdims(volume,dims=2),
        w,
        Array(a1a2_blk),
        Array(a2a3_blk), 
        permutation_blk, 
        sblk, 
        nblk
    )

end

function get_stride(v::Vector)
    i=1
    @inbounds while v[i]==v[i+1]
    # @inbounds while v[i]<=v[i+1]
        i+=1
    end
    return i
end

function sort_axes(a2a3,a1a2)
    sblk = get_stride(a1a2) # block size
    nblk = div(length(a2a3),sblk)+1 # number of blocksa1
    blk = 1:sblk
    # blk  = 1:sblk:length(a2a3)
    sorted_blk = sort(view(a2a3,blk)) # sorted block
    permutation_blk = sortperm(view(a2a3,blk)) # sorting permutations
    return sorted_blk, permutation_blk, sblk, nblk
end

function argminsorted(v)
    m = length(v)
    for i in 1:m-1
        if @inbounds v[i] < v[i+1]
            return i
        end
    end
    return m
end

function fixtensor!(C)
    dummy = zeros(2)
    i1 = [1,3]
    i2 = [7,9]
    i3 = [4,6]
    @inbounds for i in axes(C,1)
        dummy[1], dummy[2] = abs(C[i,2]-C[i,1]), abs(C[i,2]-C[i,3])
        idx = i1[argmin(dummy)]
        C[i,idx] = C[i,2] = (C[i,idx]+C[i,2])*0.5

        dummy[1], dummy[2] = abs(C[i,8]-C[i,7]), abs(C[i,8]-C[i,9])
        idx = i2[argmin(dummy)]
        C[i,8] = C[i,idx] = (C[i,8]+C[i,idx])*0.5

        dummy[1], dummy[2] = abs(C[i,4]-C[i,5]), abs(C[i,6]-C[i,5])
        idx = i3[argmin(dummy)]
        C[i,idx] = C[i,5] = (C[i,idx]+C[i,5])*0.5
    end
end

"""
    -------------------------------------------------------------------------
    Parametrisation of average inclusion shape for WEAK inclusions
    -------------------------------------------------------------------------    
    rᵢ = ζᵢ + ξᵢ * A + χᵢ * B
    Where :
        (*) r₁ = log10(a₁ / a₂) (INCLUSION FSE)
        (*) r₂ = log10(a₂ / a₃) (INCLUSION FSE)
        (*) A  = log10(a₁ / a₂) (BULK FSE)
        (*) B  = log10(a₂ / a₃) (BULK FSE)
        (*) Greek characters -> fitting coefficients 
    Input :
        (*) a1, a2, a3 -> principal semi-axes of the BULK FSE   
""" 
function fabric_parametrisation(a1::Float64, a2::Float64, a3::Float64, R1::Parameterisation, R2::Parameterisation)
    # -- Bulk semi axes ratios
    A = log10(a1/a2)
    B = log10(a2/a3)
    # -- Calculate: r₁ = log10(a₁ / a₂)
    r₁ = R1.ζ + R1.ξ*A + R1.χ*B
    # -- Calculate: r₂ = log10(a₂ / a₃)
    r₂ = R2.ζ + R2.ξ*A + R2.χ*B
    return r₁, r₂ 
end

function fittingcoefficients()
    # ==== Weak Inclusions (values from paper)
    # -- Calculate: r₁ = log10(a₁ / a₂)
    ζ =  0.015159;
    ξ =  1.1013;
    χ = -0.093104;
    R1 = Parameterisation(ζ,ξ,χ)
        
    # -- Calculate: r₂ = log10(a₂ / a₃)
    ζ =  0.0028906;
    ξ =  1.0533;
    χ =  0.23141;
    R2 = Parameterisation(ζ,ξ,χ)
    return R1,R2
end

# anisotropic_tensor(FSE::Array{FiniteStrainEllipsoid{Float64},2}, D::DEM, ::Val{Isotropic}, ipx) = 
#     StiffnessTensor([0.0 0.0], [0.0 0.0], [0.0 0.0], [0.0 0.0], [0.0 0.0], [0.0 0.0])
    
# function anisotropic_tensor(FSE::Array{FiniteStrainEllipsoid{Float64},2}, D::DEM, ::Val{Anisotropic},ipx)
function anisotropic_tensor(FSE::Array{FiniteStrainEllipsoid{Float64},2}, D::DEM, ::Val{T}, ipx) where {T}
    # -- Allocate arrays
    nu_11 = Array{Float64,2}(undef, size(FSE, 1), 7)
    nu_33 = similar(nu_11)
    nu_55 = similar(nu_11)
    nu_13 = similar(nu_11)
    nu_15 = similar(nu_11)
    nu_35 = similar(nu_11)
    v₁ = [similar(D.a1a2_blk) for _ in 1:Threads.nthreads()]
    v₂ = [similar(D.a2a3_blk) for _ in 1:Threads.nthreads()]
    max_a1a2, max_a2a3 = maximum(D.a1a2_blk), maximum(D.a2a3_blk)
    
    # -- Fitting coefficients of the axes parameterisation
    R1, R2 = fittingcoefficients()
    
    # -- Get η from data base and rotate it
    for i in eachindex(FSE)
        get_tensor_and_rotate!( nu_11, nu_33, nu_55, nu_13, nu_15, nu_35,
                                FSE[i],R1,R2,D,i,v₁,v₂, max_a1a2, max_a2a3,ipx[i])
    end

    # -- 7th ip
    nu_11[:,7] .= @. (nu_11[:,1] + nu_11[:,2] + nu_11[:,3] + nu_11[:,4] + nu_11[:,5] + nu_11[:,6])/6
    nu_33[:,7] .= @. (nu_33[:,1] + nu_33[:,2] + nu_33[:,3] + nu_33[:,4] + nu_33[:,5] + nu_33[:,6])/6
    nu_55[:,7] .= @. (nu_55[:,1] + nu_55[:,2] + nu_55[:,3] + nu_55[:,4] + nu_55[:,5] + nu_55[:,6])/6
    nu_13[:,7] .= @. (nu_13[:,1] + nu_13[:,2] + nu_13[:,3] + nu_13[:,4] + nu_13[:,5] + nu_13[:,6])/6
    nu_15[:,7] .= @. (nu_15[:,1] + nu_15[:,2] + nu_15[:,3] + nu_15[:,4] + nu_15[:,5] + nu_15[:,6])/6
    nu_35[:,7] .= @. (nu_35[:,1] + nu_35[:,2] + nu_35[:,3] + nu_35[:,4] + nu_35[:,5] + nu_35[:,6])/6

    return StiffnessTensor(nu_11, nu_33, nu_55, nu_13, nu_15, nu_35)

end ### END rotate_tensor FUNCTION #############################################

function argminsortedsecond(v)
    imin = 0

    for i in 1:length(v)-1
        if @inbounds v[i] < v[i+1]
            imin = i
        end
    end

    if imin != 0

        if  @inbounds v[imin+1] < v[imin-1]
            return imin, imin+1
        else
            return imin, imin-1
        end
    
    elseif imin == 0
        return length(v), length(v)

    elseif imin == 1
        return 1, 1
    end
end


function get_tensor_and_rotate!(nu_11, nu_33, nu_55, nu_13, nu_15, nu_35,
                                FSEᵢ,R1,R2,D,i,v₁,v₂,max_a1a2,max_a2a3, ipx)
    # Average fabric -> r₁ = log10(a1/a2) and r₂ = log10(a2/a3)
    a1 = applybounds(FSEᵢ.a1, 25.0, 1.0)
    a2 = applybounds(FSEᵢ.a2, 1.0, 0.01)

    r₁, r₂ = fabric_parametrisation(a1,
                                    1.0,
                                    a2, 
                                    R1,
                                    R2)
    nt = Threads.threadid()

    # if r₁ > max_a1a2
    #     # r₁_imin = 49
    #     r₁_imin, r₁_imin2 = 49, 49
    # else
    @inbounds for j in eachindex(D.a1a2_blk)
        v₁[nt][j] = abs(r₁-D.a1a2_blk[j])
    end
    r₁_imin = argminsorted(v₁[nt])
        # r₁_imin, r₁_imin2 = argminsortedsecond(v₁[nt])
    # end

    # if r₂ > max_a2a3
    #     r₂_imin, r₂_imin2 = 50, 50
    # else
        @inbounds for j in eachindex(D.a2a3_blk)
            v₂[nt][j] = abs(r₂-D.a2a3_blk[j])
        end
        r₂_imin = D.permutation_blk[argminsorted(v₂[nt])]
        # idx = argminsortedsecond(v₂[nt])
        # r₂_imin = D.permutation_blk[idx[1]]
        # r₂_imin2 = D.permutation_blk[idx[2]]

    # end

    im = D.sblk*(r₁_imin-1) + r₂_imin
    # Allocate stiffness tensor
    C = @SMatrix [D.𝓒[im, 1]  D.𝓒[im, 7]  D.𝓒[im, 8]  0          0           0
                  D.𝓒[im, 7]  D.𝓒[im, 2]  D.𝓒[im, 9]  0          0           0
                  D.𝓒[im, 8]  D.𝓒[im, 9]  D.𝓒[im, 3]  0          0           0
                  0           0           0           D.𝓒[im, 4] 0           0 
                  0           0           0           0          D.𝓒[im, 5]  0
                  0           0           0           0          0           D.𝓒[im, 6]]
    ## =================================================================

    ## ROTATE VISCOUS TENSOR ===========================================
    a = atand(FSEᵢ.y1, FSEᵢ.x1)
    R = @SMatrix [cosd(a)  0   sind(a)
                   0       1   0
                 -sind(a)  0   cosd(a)]
    # ----- Rotate tensor (fast version derived from symbolic calculus)
    η11,η33,η55,η13,η15,η35 = directRotation2D(R,C)
    # dummy = (abs(η15)+abs(η35))*0.5
    @fastmath begin
        nu_11[i] = η11
        nu_33[i] = η33
        nu_55[i] = max(η55, 0.27)
        # nu_55[i] = η55
        nu_13[i] = η13
        nu_15[i] = η15
        nu_35[i] = η35
        # nu_15[i] = sign(η15)*dummy
        # nu_35[i] = sign(η35)*dummy

        # nu_11[i] = η11 - η13
        # nu_33[i] = η33 - η13
        # # nu_55[i] = max(η55, 0.1)
        # nu_55[i] = η55
        # nu_13[i] = η13*0
        # nu_15[i] = η15
        # nu_35[i] = η35
    end
    # ==================================================================
end

function unrotate_anisotropic_tensor(𝓒,vx1,vx2,vy1,vy2)
    η11 = 𝓒.η11[:,1:6]
    η33 = 𝓒.η33[:,1:6]
    η55 = 𝓒.η55[:,1:6]
    η13 = 𝓒.η13[:,1:6]
    η15 = 𝓒.η15[:,1:6]
    η35 = 𝓒.η35[:,1:6]
    
    # -- Get η from data base and rotate it
    for i in CartesianIndices(η11)
        # Allocate stiffness tensor
        C = @SMatrix [ η11[i]  0  η13[i]  0  η15[i]  0
                       0       0  0       0  0       0
                       η13[i]  0  η33[i]  0  η35[i]  0
                       0       0  0       0  0       0 
                       η15[i]  0  η35[i]  0  η55[i]  0
                       0       0  0       0  0       0]
        ## =================================================================

        ## ROTATE VISCOUS TENSOR ===========================================
        R = @SMatrix [vx1[i]  0.0   vx2[i]
                      0.0     1.0   0.0
                      vy1[i]  0.0   vy2[i]]
        # ----- Rotate tensor (fast version derived from symbolic calculus)
        # nu_11[i], nu_33[i], nu_55[i], nu_13[i], nu_15[i], nu_35[i] = directRotation2D(R,C)
        n11,n33,n55,n13,n15,n35 = directRotation2D(inv(R),C)
        𝓒.η11[i] = n11
        𝓒.η33[i] = n33
        𝓒.η55[i] = n55
        𝓒.η13[i] = n13
        𝓒.η15[i] = n15
        𝓒.η35[i] = n35
    end
    
    return 𝓒

end ### END rotate_tensor FUNCTION #############################################

function anisotropic_tensor_blks(FSE::Array{FiniteStrainEllipsoid{Float64},2}, D::DEM)
    # -- Allocate arrays
    n = size(FSE, 1)
    nu0 = Vector{Float64}(undef, 9)
    nu_interp = fill(0.0,6,6)
    nu_11 = Array{Float64,2}(undef, n, 6)
    nu_33 = similar(nu_11)
    nu_55 = similar(nu_11)
    nu_13 = similar(nu_11)
    nu_15 = similar(nu_11)
    nu_35 = similar(nu_11)

    # -- Fitting coefficients of the axes parameterisation
    R1,R2 = fittingcoefficients()

    # -- threading blocks
    nblk = Threads.nthreads()
    sblk = div(length(FSE),nblk) 
    blk = 1:sblk

    # -- Get η from data base and rotate it
    @sync begin
        for i in 1:nblk-1
            Threads.@spawn get_tensor_and_rotate_blk!( nu_11, nu_33, nu_55, nu_13, nu_15, nu_35,
                                FSE,R1,R2,D, blk.*(1 + (i-1)))
        end
        get_tensor_and_rotate_blk!( nu_11, nu_33, nu_55, nu_13, nu_15, nu_35,
                                    FSE,R1,R2,D, sblk*(nblk-1):length(FSE))
    end

    return StiffnessTensor(nu_11, nu_33, nu_55, nu_13, nu_15, nu_35)

end ### END rotate_tensor FUNCTION #############################################


function get_tensor_and_rotate_blk!(nu_11, nu_33, nu_55, nu_13, nu_15, nu_35,
    FSE,R1,R2,D,iblk)
    @inbounds for i in iblk
        ## GET VISCOUS TENSOR ==============================================
        # Average fabric -> r₁ = log10(a1/a2) and r₂ = log10(a2/a3)
        F = FSE[i]
        r₁, r₂ = fabric_parametrisation(F.a1, 
                                        1.0, 
                                        F.a2, 
                                        R1, 
                                        R2)

        v₁ = abs.(r₁.-D.a1a2_blk)
        v₂ = abs.(r₂.-D.a2a3_blk)
        r₁_imin = argminsorted(v₁)
        r₂_imin = D.permutation_blk[argminsorted(v₂)]
        im = D.sblk*(r₁_imin-1) + r₂_imin
        𝓒 = view(D.𝓒,im,:)
        # Allocate stiffness tensor
        C = @SMatrix [𝓒[1]  𝓒[9]  𝓒[8]  0     0    0
                      𝓒[9]  𝓒[2]  𝓒[7]  0     0    0
                      𝓒[8]  𝓒[7]  𝓒[3]  0     0    0
                      0     0     0     𝓒[4] 0     0 
                      0     0     0     0    𝓒[5]  0
                      0     0     0     0    0     𝓒[6]]
        ## =================================================================

        ## ROTATE VISCOUS TENSOR ===========================================
        R = @SMatrix [F.x1  0   F.x2
                      0          1   0
                      F.y1  0   F.y2]
        # ----- Rotate tensor  
        # Cr = TensorRotation(R,C)
        nu_11[i], nu_33[i], nu_55[i], nu_13[i], nu_15[i], nu_35[i] = directRotation2D(R,C)
        # ==================================================================
    end
end

function custommin(a::AbstractArray, b::Float64)
    imin = 1 # index of min value
    vmin = (a[1] - b)^2 # min value

    @inbounds for i = 2:length(a)
        d = (a[i] - b)^2 
        if vmin > d
            imin = i
            vmin = d
        end

    end
    return imin
end

"""
    Allocate K matrix (based on Bowers 'Applied Mechanics of Solids', Chapter 3]
"""
function getK(R) 
    # 11 block
    K11 = R[1,1]*R[1,1]
    K12 = R[1,2]*R[1,2]
    K13 = R[1,3]*R[1,3]
    K21 = R[2,1]*R[2,1]
    K22 = R[2,2]*R[2,2]
    K23 = R[2,3]*R[2,3]
    K31 = R[3,1]*R[3,1]
    K32 = R[3,2]*R[3,2]
    K33 = R[3,3]*R[3,3]
    # 12 block
    K14 = 2*R[1,2]*R[1,3]
    K15 = 2*R[1,3]*R[1,1]
    K16 = 2*R[1,1]*R[1,2]
    K24 = 2*R[2,2]*R[2,3]
    K25 = 2*R[2,3]*R[2,1]
    K26 = 2*R[2,1]*R[2,2]
    K34 = 2*R[3,2]*R[3,3]
    K35 = 2*R[3,3]*R[3,1]
    K36 = 2*R[3,1]*R[3,2]
    # 21 block
    K41 =  R[2,1]*R[3,1] 
    K42 =  R[2,2]*R[3,2] 
    K43 =  R[2,3]*R[3,3]
    K51 =  R[3,1]*R[1,1] 
    K52 =  R[3,2]*R[1,2] 
    K53 =  R[3,3]*R[1,3] 
    K61 =  R[1,1]*R[2,1] 
    K62 =  R[1,2]*R[2,2] 
    K63 =  R[1,3]*R[2,3] 
    # 22 block
    K44 =  R[2,2]*R[3,3]+R[2,3]*R[3,2] 
    K54 =  R[2,3]*R[3,1]+R[2,1]*R[3,3]     
    K64 =  R[2,1]*R[3,2]+R[2,2]*R[3,1]
    K45 =  R[3,2]*R[1,3]+R[3,3]*R[1,2]   
    K55 =  R[3,3]*R[1,1]+R[3,1]*R[1,3]     
    K65 =  R[3,1]*R[1,2]+R[3,2]*R[1,1]
    K46 =  R[1,2]*R[2,3]+R[1,3]*R[2,2]   
    K56 =  R[1,3]*R[2,1]+R[1,1]*R[2,3]     
    K66 =  R[1,1]*R[2,2]+R[1,2]*R[2,1]

    K = @SMatrix [K11 K12 K13 K14 K15 K16
                  K21 K22 K23 K24 K25 K26
                  K31 K32 K33 K34 K35 K36
                  K41 K42 K43 K44 K45 K46
                  K51 K52 K53 K54 K55 K56
                  K61 K62 K63 K64 K65 K66]

end

function TensorRotation(R,C)
    K = getK(R) # -> rotation 6x6 matrix
    K*C*K' # -> rotated 4th rank tensor
end

function directRotation2D(R,C)
    ## Tensor coefficients
    a = C[1,1]
    # b = C[2,2]
    c = C[3,3]
    # d = C[4,4]
    e = C[5,5]
    # f = C[6,6]
    # g = C[1,2]
    h = C[1,3]
    # l = C[2,3]
    ## Rotation matrix
    x1 = R[1,1]
    x2 = R[1,3]
    y1 = R[3,1]
    y2 = R[3,3]
    ## Rotated components
    n11 = (c * x2 ^ 2 + h * x1 ^ 2 + 4 * e * x1 ^ 2) * x2 ^ 2 + (a * x1 ^ 2 + h * x2 ^ 2) * x1 ^ 2;
    n33 = (c * y2 ^ 2 + h * y1 ^ 2 + 4 * e * y1 ^ 2) * y2 ^ 2 + (a * y1 ^ 2 + h * y2 ^ 2) * y1 ^ 2;
    n55 = (c * x2 * y2 + h * x1 * y1) * x2 * y2 + (x1 * y2 + x2 * y1) ^ 2 * e + (a * x1 * y1 + h * x2 * y2) * x1 * y1;
    n13 = ((c * x2 ^ 2 + h * x1 ^ 2) * y2 + 4 * e * x1 * x2 * y1) * y2 + (a * x1 ^ 2 + h * x2 ^ 2) * y1 ^ 2;
    n15 = ((c * x2 ^ 2 + h * x1 ^ 2) * y2 + 2 * (x1 * y2 + x2 * y1) * e * x1) * x2 + (a * x1 ^ 2 + h * x2 ^ 2) * x1 * y1;
    n35 = ((c * y2 ^ 2 + h * y1 ^ 2) * x2 + 2 * (x1 * y2 + x2 * y1) * e * y1) * y2 + (a * y1 ^ 2 + h * y2 ^ 2) * x1 * y1;

    return n11,n33,n55,n13,n15,n35
end


function directRotation(R,C)
    @fastmath begin
        ## Tensor coefficients
        a = C[1,1]
        b = C[2,2]
        c = C[3,3]
        d = C[4,4]
        e = C[5,5]
        f = C[6,6]
        g = C[1,2]
        h = C[1,3]
        l = C[2,3]
        ## Rotation matrix
        x1 = R[1,1]
        x2 = R[1,2]
        x3 = R[1,3]
        y1 = R[2,1]
        y2 = R[2,2]
        y3 = R[2,3]
        z1 = R[3,1]
        z2 = R[3,2]
        z3 = R[3,3]
        ## Rotated components
        n11 = 4 * ((e * x3 ^ 2 + f * x2 ^ 2) * x1 ^ 2 + d * x2 ^ 2 * x3 ^ 2) + (h * x1 ^ 2 + l * x2 ^ 2 + c * x3 ^ 2) * x3 ^ 2 + (g * x2 ^ 2 + h * x3 ^ 2 + a * x1 ^ 2) * x1 ^ 2 + (g * x1 ^ 2 + l * x3 ^ 2 + b * x2 ^ 2) * x2 ^ 2
        n13 = 4 * ((e * x3 * z3 + f * x2 * z2) * x1 * z1 + d * x2 * x3 * z2 * z3) + (h * x1 ^ 2 + l * x2 ^ 2 + c * x3 ^ 2) * z3 ^ 2 + (g * x2 ^ 2 + h * x3 ^ 2 + a * x1 ^ 2) * z1 ^ 2 + (g * x1 ^ 2 + l * x3 ^ 2 + b * x2 ^ 2) * z2 ^ 2
        n15 = 2 * (((x1 * z3 + x3 * z1) * e * x1 + (y1 * z3 + y3 * z1) * d * x2) * x3 + (x1 * y3 + x3 * y1) * f * x1 * x2) + (h * x1 ^ 2 + l * x2 ^ 2 + c * x3 ^ 2) * x3 * z3 + (g * x2 ^ 2 + h * x3 ^ 2 + a * x1 ^ 2) * x1 * z1 + (g * x1 ^ 2 + l * x3 ^ 2 + b * x2 ^ 2) * x2 * z2
        n33 = 4 * ((e * z3 ^ 2 + f * z2 ^ 2) * z1 ^ 2 + d * z2 ^ 2 * z3 ^ 2) + (h * z1 ^ 2 + l * z2 ^ 2 + c * z3 ^ 2) * z3 ^ 2 + (g * z2 ^ 2 + h * z3 ^ 2 + a * z1 ^ 2) * z1 ^ 2 + (g * z1 ^ 2 + l * z3 ^ 2 + b * z2 ^ 2) * z2 ^ 2
        n35 = 2 * (((x1 * z3 + x3 * z1) * e * z1 + (y1 * z3 + y3 * z1) * d * z2) * z3 + (x1 * y3 + x3 * y1) * f * z1 * z2) + (h * z1 ^ 2 + l * z2 ^ 2 + c * z3 ^ 2) * x3 * z3 + (g * z2 ^ 2 + h * z3 ^ 2 + a * z1 ^ 2) * x1 * z1 + (g * z1 ^ 2 + l * z3 ^ 2 + b * z2 ^ 2) * x2 * z2
        n55 = (x1 * z3 + x3 * z1) ^ 2 * e + (y1 * z3 + y3 * z1) ^ 2 * d + (x1 * y3 + x3 * y1) ^ 2 * f + (h * x1 * z1 + l * x2 * z2 + c * x3 * z3) * x3 * z3 + (g * x2 * z2 + h * x3 * z3 + a * x1 * z1) * x1 * z1 + (g * x1 * z1 + l * x3 * z3 + b * x2 * z2) * x2 * z2
    end

    return n11,n33,n55,n13,n15,n35

end

function directRotation!(n, R, C)
    ## Tensor
    a = C[1,1]
    b = C[2,2]
    c = C[3,3]
    d = C[4,4]
    e = C[5,5]
    f = C[6,6]
    g = C[1,2]
    h = C[1,3]
    l = C[2,3]
    ## Rotation matrix
    x1 = R[1,1]
    x2 = R[1,2]
    x3 = R[1,3]
    y1 = R[2,1]
    y2 = R[2,2]
    y3 = R[2,3]
    z1 = R[3,1]
    z2 = R[3,2]
    z3 = R[3,3]
    ## Rotated components
    n[1,1] = 4 * ((e * x3 ^ 2 + f * x2 ^ 2) * x1 ^ 2 + d * x2 ^ 2 * x3 ^ 2) + (h * x1 ^ 2 + l * x2 ^ 2 + c * x3 ^ 2) * x3 ^ 2 + (g * x2 ^ 2 + h * x3 ^ 2 + a * x1 ^ 2) * x1 ^ 2 + (g * x1 ^ 2 + l * x3 ^ 2 + b * x2 ^ 2) * x2 ^ 2
    n[1,2] = n[1,2] = 4 * ((e * x3 * y3 + f * x2 * y2) * x1 * y1 + d * x2 * x3 * y2 * y3) + (h * x1 ^ 2 + l * x2 ^ 2 + c * x3 ^ 2) * y3 ^ 2 + (g * x2 ^ 2 + h * x3 ^ 2 + a * x1 ^ 2) * y1 ^ 2 + (g * x1 ^ 2 + l * x3 ^ 2 + b * x2 ^ 2) * y2 ^ 2
    n[2,2] = 4 * ((e * y3 ^ 2 + f * y2 ^ 2) * y1 ^ 2 + d * y2 ^ 2 * y3 ^ 2) + (h * y1 ^ 2 + l * y2 ^ 2 + c * y3 ^ 2) * y3 ^ 2 + (g * y2 ^ 2 + h * y3 ^ 2 + a * y1 ^ 2) * y1 ^ 2 + (g * y1 ^ 2 + l * y3 ^ 2 + b * y2 ^ 2) * y2 ^ 2
    n[1,3] = n[1,3] = 4 * ((e * x3 * z3 + f * x2 * z2) * x1 * z1 + d * x2 * x3 * z2 * z3) + (h * x1 ^ 2 + l * x2 ^ 2 + c * x3 ^ 2) * z3 ^ 2 + (g * x2 ^ 2 + h * x3 ^ 2 + a * x1 ^ 2) * z1 ^ 2 + (g * x1 ^ 2 + l * x3 ^ 2 + b * x2 ^ 2) * z2 ^ 2
    n[2,3] = n[3,2] = 4 * ((e * y3 * z3 + f * y2 * z2) * y1 * z1 + d * y2 * y3 * z2 * z3) + (h * y1 ^ 2 + l * y2 ^ 2 + c * y3 ^ 2) * z3 ^ 2 + (g * y2 ^ 2 + h * y3 ^ 2 + a * y1 ^ 2) * z1 ^ 2 + (g * y1 ^ 2 + l * y3 ^ 2 + b * y2 ^ 2) * z2 ^ 2
    n[3,3] = 4 * ((e * z3 ^ 2 + f * z2 ^ 2) * z1 ^ 2 + d * z2 ^ 2 * z3 ^ 2) + (h * z1 ^ 2 + l * z2 ^ 2 + c * z3 ^ 2) * z3 ^ 2 + (g * z2 ^ 2 + h * z3 ^ 2 + a * z1 ^ 2) * z1 ^ 2 + (g * z1 ^ 2 + l * z3 ^ 2 + b * z2 ^ 2) * z2 ^ 2
    n[1,4] = n[1,4] = 2 * (((x2 * z3 + x3 * z2) * e * x1 + (y2 * z3 + y3 * z2) * d * x2) * x3 + (x2 * y3 + x3 * y2) * f * x1 * x2) + (h * x1 ^ 2 + l * x2 ^ 2 + c * x3 ^ 2) * y3 * z3 + (g * x2 ^ 2 + h * x3 ^ 2 + a * x1 ^ 2) * y1 * z1 + (g * x1 ^ 2 + l * x3 ^ 2 + b * x2 ^ 2) * y2 * z2
    n[2,4] = n[4,2] = 2 * (((x2 * z3 + x3 * z2) * e * y1 + (y2 * z3 + y3 * z2) * d * y2) * y3 + (x2 * y3 + x3 * y2) * f * y1 * y2) + (h * y1 ^ 2 + l * y2 ^ 2 + c * y3 ^ 2) * y3 * z3 + (g * y2 ^ 2 + h * y3 ^ 2 + a * y1 ^ 2) * y1 * z1 + (g * y1 ^ 2 + l * y3 ^ 2 + b * y2 ^ 2) * y2 * z2
    n[3,4] = n[4,3] = 2 * (((x2 * z3 + x3 * z2) * e * z1 + (y2 * z3 + y3 * z2) * d * z2) * z3 + (x2 * y3 + x3 * y2) * f * z1 * z2) + (h * z1 ^ 2 + l * z2 ^ 2 + c * z3 ^ 2) * y3 * z3 + (g * z2 ^ 2 + h * z3 ^ 2 + a * z1 ^ 2) * y1 * z1 + (g * z1 ^ 2 + l * z3 ^ 2 + b * z2 ^ 2) * y2 * z2
    n[4,4] = (x2 * z3 + x3 * z2) ^ 2 * e + (y2 * z3 + y3 * z2) ^ 2 * d + (x2 * y3 + x3 * y2) ^ 2 * f + (h * y1 * z1 + l * y2 * z2 + c * y3 * z3) * y3 * z3 + (g * y2 * z2 + h * y3 * z3 + a * y1 * z1) * y1 * z1 + (g * y1 * z1 + l * y3 * z3 + b * y2 * z2) * y2 * z2
    n[1,5] = n[5,1] = 2 * (((x1 * z3 + x3 * z1) * e * x1 + (y1 * z3 + y3 * z1) * d * x2) * x3 + (x1 * y3 + x3 * y1) * f * x1 * x2) + (h * x1 ^ 2 + l * x2 ^ 2 + c * x3 ^ 2) * x3 * z3 + (g * x2 ^ 2 + h * x3 ^ 2 + a * x1 ^ 2) * x1 * z1 + (g * x1 ^ 2 + l * x3 ^ 2 + b * x2 ^ 2) * x2 * z2
    n[2,5] = n[5,2] = 2 * (((x1 * z3 + x3 * z1) * e * y1 + (y1 * z3 + y3 * z1) * d * y2) * y3 + (x1 * y3 + x3 * y1) * f * y1 * y2) + (h * y1 ^ 2 + l * y2 ^ 2 + c * y3 ^ 2) * x3 * z3 + (g * y2 ^ 2 + h * y3 ^ 2 + a * y1 ^ 2) * x1 * z1 + (g * y1 ^ 2 + l * y3 ^ 2 + b * y2 ^ 2) * x2 * z2
    n[3,5] = n[5,3] = 2 * (((x1 * z3 + x3 * z1) * e * z1 + (y1 * z3 + y3 * z1) * d * z2) * z3 + (x1 * y3 + x3 * y1) * f * z1 * z2) + (h * z1 ^ 2 + l * z2 ^ 2 + c * z3 ^ 2) * x3 * z3 + (g * z2 ^ 2 + h * z3 ^ 2 + a * z1 ^ 2) * x1 * z1 + (g * z1 ^ 2 + l * z3 ^ 2 + b * z2 ^ 2) * x2 * z2
    n[4,5] = n[5,4] = (x1 * z3 + x3 * z1) * (x2 * z3 + x3 * z2) * e + (y1 * z3 + y3 * z1) * (y2 * z3 + y3 * z2) * d + (x1 * y3 + x3 * y1) * (x2 * y3 + x3 * y2) * f + (h * y1 * z1 + l * y2 * z2 + c * y3 * z3) * x3 * z3 + (g * y2 * z2 + h * y3 * z3 + a * y1 * z1) * x1 * z1 + (g * y1 * z1 + l * y3 * z3 + b * y2 * z2) * x2 * z2
    n[5,5] = (x1 * z3 + x3 * z1) ^ 2 * e + (y1 * z3 + y3 * z1) ^ 2 * d + (x1 * y3 + x3 * y1) ^ 2 * f + (h * x1 * z1 + l * x2 * z2 + c * x3 * z3) * x3 * z3 + (g * x2 * z2 + h * x3 * z3 + a * x1 * z1) * x1 * z1 + (g * x1 * z1 + l * x3 * z3 + b * x2 * z2) * x2 * z2
    n[1,6] = n[6,1] = 2 * (((x1 * z2 + x2 * z1) * e * x1 + (y1 * z2 + y2 * z1) * d * x2) * x3 + (x1 * y2 + x2 * y1) * f * x1 * x2) + (h * x1 ^ 2 + l * x2 ^ 2 + c * x3 ^ 2) * x3 * y3 + (g * x2 ^ 2 + h * x3 ^ 2 + a * x1 ^ 2) * x1 * y1 + (g * x1 ^ 2 + l * x3 ^ 2 + b * x2 ^ 2) * x2 * y2
    n[2,6] = n[6,2] = 2 * (((x1 * z2 + x2 * z1) * e * y1 + (y1 * z2 + y2 * z1) * d * y2) * y3 + (x1 * y2 + x2 * y1) * f * y1 * y2) + (h * y1 ^ 2 + l * y2 ^ 2 + c * y3 ^ 2) * x3 * y3 + (g * y2 ^ 2 + h * y3 ^ 2 + a * y1 ^ 2) * x1 * y1 + (g * y1 ^ 2 + l * y3 ^ 2 + b * y2 ^ 2) * x2 * y2
    n[3,6] = n[6,3] = 2 * (((x1 * z2 + x2 * z1) * e * z1 + (y1 * z2 + y2 * z1) * d * z2) * z3 + (x1 * y2 + x2 * y1) * f * z1 * z2) + (h * z1 ^ 2 + l * z2 ^ 2 + c * z3 ^ 2) * x3 * y3 + (g * z2 ^ 2 + h * z3 ^ 2 + a * z1 ^ 2) * x1 * y1 + (g * z1 ^ 2 + l * z3 ^ 2 + b * z2 ^ 2) * x2 * y2
    n[4,6] = n[6,4] = (x1 * z2 + x2 * z1) * (x2 * z3 + x3 * z2) * e + (y1 * z2 + y2 * z1) * (y2 * z3 + y3 * z2) * d + (x1 * y2 + x2 * y1) * (x2 * y3 + x3 * y2) * f + (h * y1 * z1 + l * y2 * z2 + c * y3 * z3) * x3 * y3 + (g * y2 * z2 + h * y3 * z3 + a * y1 * z1) * x1 * y1 + (g * y1 * z1 + l * y3 * z3 + b * y2 * z2) * x2 * y2
    n[5,6] = n[6,5] = (x1 * z2 + x2 * z1) * (x1 * z3 + x3 * z1) * e + (y1 * z2 + y2 * z1) * (y1 * z3 + y3 * z1) * d + (x1 * y2 + x2 * y1) * (x1 * y3 + x3 * y1) * f + (h * x1 * z1 + l * x2 * z2 + c * x3 * z3) * x3 * y3 + (g * x2 * z2 + h * x3 * z3 + a * x1 * z1) * x1 * y1 + (g * x1 * z1 + l * x3 * z3 + b * x2 * z2) * x2 * y2
    n[6,6] = (x1 * z2 + x2 * z1) ^ 2 * e + (y1 * z2 + y2 * z1) ^ 2 * d + (x1 * y2 + x2 * y1) ^ 2 * f + (h * x1 * y1 + l * x2 * y2 + c * x3 * y3) * x3 * y3 + (g * x2 * y2 + h * x3 * y3 + a * x1 * y1) * x1 * y1 + (g * x1 * y1 + l * x3 * y3 + b * x2 * y2) * x2 * y2
end

tensor2voigt(𝓒) = [[𝓒.η11[i] 𝓒.η13[i] 𝓒.η15[i] 
                    𝓒.η13[i] 𝓒.η33[i] 𝓒.η35[i]
                    𝓒.η15[i] 𝓒.η35[i] 𝓒.η55[i]] for i in CartesianIndices(𝓒.η11)]