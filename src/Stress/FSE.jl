struct FiniteStrainEllipsoid{T}
    x1::T
    x2::T
    y1::T
    y2::T
    a1::T
    a2::T
end

rebuild_FSE(vx1, vx2, vy1, vy2, a1, a2) = [FiniteStrainEllipsoid(vx1[i], vx2[i], vy1[i], vy2[i], a1[i], a2[i]) for i in CartesianIndices(a1)]

function isotropic_lithosphere!(F,idx)
    Is = @SMatrix [1.0 0.0; 0.0 1.0]
    Threads.@threads for i in idx
        @inbounds F[i] = Is
    end
end

function healing(F, FSE)
    Is = @SMatrix [1.0 0.0; 0.0 1.0]
    Threads.@threads for i in CartesianIndices(F)
        @inbounds if FSE[i].a1./FSE[i].a2 > 1e2
            F[i] = Is
        end
    end
    return F
end


function normalize_F!(F; ϵ = 1e50)
    normalizer = abs(maximum(F[1])) # make sure we take the absolute value, don't want to change signs
    order = log10(normalizer) # need to check only one component, as the rest will havea similar exponent
    if order > ϵ
        normalizer = 1/normalizer
        Threads.@threads for i in eachindex(F)
            @inbounds @fastmath F[i] .*= normalizer
        end
    end
end

function normalize_F(F; ϵ = 1e30)
    Fmax = abs(maximum(F)) # take absolute value because we do not want to change the sign of Fij
    if Fmax > ϵ
        return F./Fmax
    else 
        return F
    end
end

function getFSE(F, FSE)
    # F can grow A LOT in long computations, eventually overflowing at ~1e309
    # Thus we need to normalize F from time to time. We normalize F ∈ Ω w.r.t.
    # the same number (max(F[1])), otherwise a heterogeneous normalization 
    # will screw up the particles interpolation 
    normalize_F!(F) 
    Threads.@threads for iel in CartesianIndices(F)
        @inbounds FSE[iel] = _FSE(F[iel])
    end
    FSE
end 

function _FSE(Fi)
    
    # Compute FSE
    # Fi = normalize_F(F) # normalize F if its too big, otherwise F*F' will be Inf
    L = Fi * Fi' # left-strecth tensor
    # eigval, evect = eigen(Fi * Fi') # get length of FSE semi-axes and orientation
    # imax, imin = eigval_order(eigval) # get the right order of the semi-axis length
    eigval, evect, imax, imin = _eigen(L) # get length of FSE semi-axes and orientation
                                          # and the right order of eigvals and eigvects

    # Fill FSE
    FiniteStrainEllipsoid(
        evect[1,imax], # vx1
        evect[1,imin], # vx2
        evect[2,imax], # vy1
        evect[2,imin], # vy2
        √(abs(eigval[imax])), # a1 
        √(abs(eigval[imin])), # a2
    )

end

eigval_order(eigval) = ifelse(
    @inbounds(eigval[2] > eigval[1]),
    (2, 1),
    (1, 2)
)

function getFSE_healing(F, FSE; ϵ = 1e3)
    Threads.@threads for iel in eachindex(F)
        healing_FSE!(FSE, F, iel, ϵ)
    end
    FSE, F
end 

function healing_FSE!(FSE, F, iel, ϵ)
    
    # Compute FSE
    Fi = normalize_F(F[iel]) # normalize F if its too big, otherwise F*F' will be Inf
    eigval, evect = eigen(Fi * Fi') # get length of FSE semi-axes and orientation
    imax, imin = eigval_order(eigval) # get the right order of the semi-axis length

    a1 = √(abs(eigval[imax]))
    a2 = √(abs(eigval[imin]))
    @inbounds if a1/a2 < ϵ
        # Fill FSE
        FSE[iel] = FiniteStrainEllipsoid(
            evect[1,imax]::Float64, # vx1
            evect[1,imin]::Float64, # vx2
            evect[2,imax]::Float64, # vy1
            evect[2,imin]::Float64, # vy2
            a1, # a1 
            a2, # a2
        )

    else
        F[iel] = @SMatrix [1.0 0.0; 0.0 1.0]
        # Fill FSE
        FSE[iel] = FiniteStrainEllipsoid(
            1.0, # vx1
            0.0, # vx2
            0.0, # vy1
            1.0, # vy2
            1.0, # a1 
            1.0, # a2
        )

    end

end

function getFSE_annealing!(F, FSE, annealing)
    Threads.@threads for iel in eachindex(F)
        @inbounds FSE[iel] = _FSE_annealing(F[iel], annealing)
    end
end

function _FSE_annealing(
    F, 
    s
)
    
    # Compute FSE
    Fi = normalize_F(F) # normalize F if its too big, otherwise F*F' will be Inf
    L = Fi * Fi' # left-strecth tensor
    # eigval, evect = eigen(Fi * Fi') # get length of FSE semi-axes and orientation
    # imax, imin = eigval_order(eigval) # get the right order of the semi-axis length
    eigval, evect, imax, imin = _eigen(L) # get length of FSE semi-axes and orientation
                                          # and the right order of eigvals and eigvects
                                        
    # principal semi-axes
    a1, a2 = √(abs(eigval[imax])),  √(abs(eigval[imin]))
    # eigenvectors
    λ11, λ12, λ21, λ22 = evect[1,imax], evect[2,imax], evect[1,imin], evect[2,imin]
    # unstretch semi-axes due to annealing
    a1u, a2u = unstretch_axes(a1, a2, s)
    # Fu = recover_F(Fi, L, λ11, λ12, λ21, λ22, a1u, a2u)
    # Fu = recover_F(Fi, L, λ11, λ12, λ21, λ22, a1, a2)

    # Fill FSE
    FSE = FiniteStrainEllipsoid(
        λ11, # vx1
        λ21, # vx2
        λ12, # vy1
        λ22, # vy2
        a1u, 
        a2u,
    )

    return FSE

end

function _eigen(A)
    eigval::SVector{2, Float64}, evect::SMatrix{2, 2, Float64} = eigen(A) # get length of FSE semi-axes and orientation
    # eigval, evect = eigen(A) # get length of FSE semi-axes and orientation
    imax, imin = eigval_order(eigval) # get the right order of the semi-axis length
    return eigval, evect, imax, imin
end

function volume_integral(V, EL2NOD, θ, r)
    ## MODEL PARAMETERS
    nip = 6
    nnodel = size(EL2NOD,1)
    nnodel = 6
    nel = size(EL2NOD,2)
    
    ## INTEGRATION POINTS & DERIVATIVES wrt LOCAL COORDINATES   
    ni, nn, nn3 = Val(nip), Val(nnodel), Val(3)
    N, ∇N,_ , w_ip = _get_SF(ni,nn)
    N3, b, dN3ds,c = _get_SF(ni,nn3)

    ## LOOP OVER ELEMENETS
    integral = zeros(nel)
    Threads.@threads for iel in 1:nel
        #= NOTE: For triangular elements with non-curved edges the Jacobian is
              the same for all integration points (i.e. calculated once
              before the integration loop). Further, linear 3-node shape are
              sufficient to calculate the Jacobian. =#

        # Polar coordinates of element nodes
        θ_el = @SVector [θ[i,iel] for i in 1:3]
        r_el = @SVector [r[i,iel] for i in 1:3]
        iel_coords = SMatrix{3,2}([θ_el r_el])
       
        # Jacobian n. 1 (p:=polar, l:=local): reference element --> current element
        J_pl = dN3ds*iel_coords
        detJ_pl = det(J_pl) # fast |-> unrolled
        
        # Jacobian ∂ξ∂θ to transform local (ξ, η) into global (θ,r) derivatives
        #     ∂ξ∂θ = [ R_31    -R_21
        #             -Th_31    Th_21] / detJa_PL
        # ∂ξ∂θ = @SMatrix [(r_el[3]-r_el[1])/detJ_pl -(r_el[2]-r_el[1])/detJ_pl
        #                 -(θ_el[3]-θ_el[1])/detJ_pl  (θ_el[2]-θ_el[1])/detJ_pl]
    
        idx = view(EL2NOD, 1:6, iel)
        V_el = SVector{6}(view(V, idx))

        ## INTEGRATION LOOP
        @inbounds for ip in 1:nip
            # Polar coordinates of the integration points
            r_ip = mydot(r_el,N3[ip])
            
            # Integration weight
            ω = r_ip*detJ_pl*w_ip[ip]
            
            # field at integration point
            V_ip = mydot(V_el,N[ip])
            
            # Integral reduction
            integral[iel] += V_ip*ω 
        end
        
    end # end block loop
    
    return sum(integral)
    
end # END OF ASSEMBLY FUNCTION

function volume_integral_cartesian(V, gr)
    ## MODEL PARAMETERS
    EL2NOD = gr.e2n
    x = gr.x[gr.e2n]
    z = gr.z[gr.e2n]
    nip = 6
    nnodel = size(EL2NOD,1)
    nnodel = 6
    nel = size(EL2NOD,2)
    
    ## INTEGRATION POINTS & DERIVATIVES wrt LOCAL COORDINATES   
    ni, nn, nn3 = Val(nip), Val(nnodel), Val(3)
    N, ∇N,_ , w_ip = _get_SF(ni,nn)
    _, b, dN3ds,c = _get_SF(ni,nn3)

    ## LOOP OVER ELEMENETS
    integral = zeros(nel)
    Threads.@threads for iel in 1:nel
        #= NOTE: For triangular elements with non-curved edges the Jacobian is
              the same for all integration points (i.e. calculated once
              before the integration loop). Further, linear 3-node shape are
              sufficient to calculate the Jacobian. =#

        # Polar coordinates of element nodes
        x_el = @SVector [x[i,iel] for i in 1:3]
        z_el = @SVector [z[i,iel] for i in 1:3]
        iel_coords = SMatrix{3,2}([x_el z_el])
       
        # Jacobian n. 1 (p:=polar, l:=local): reference element --> current element
        J_pl = dN3ds*iel_coords
        detJ_pl = det(J_pl) # fast |-> unrolled
        
        idx = view(EL2NOD, 1:6, iel)
        V_el = SVector{6}(view(V, idx))

        ## INTEGRATION LOOP
        @inbounds for ip in 1:nip                     
            # Integration weight
            ω = detJ_pl*w_ip[ip]
            
            # field at integration point
            V_ip = mydot(V_el, N[ip])
            # Integral reduction
            integral[iel] += V_ip*ω 
        end
        
    end # end block loop
    
    return sum(integral)
    
end # END OF ASSEMBLY FUNCTION
