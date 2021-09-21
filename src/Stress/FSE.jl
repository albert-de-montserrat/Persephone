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

function healing!(F, FSE)
    Is = @SMatrix [1.0 0.0; 0.0 1.0]
    Threads.@threads for i in axes(F, 2)
        @inbounds if FSE[i].a1./FSE[i].a2 > 1e2
            F[i] = Is
        end
    end
end

function getFSE(F, FSE)
    @batch for iel in eachindex(F)
        _FSE!(FSE, F, iel)
    end
    FSE
end 

function _FSE!(FSE, F, iel)
    
    # Compute FSE
    eigval, evect = eigen(F[iel] * F[iel]')
    if eigval[2] > eigval[1]
        imax, imin = 2, 1
    else
        imax, imin = 1, 2
    end

    # Fill FSE
    @inbounds FSE[iel] = FiniteStrainEllipsoid(
        evect[1,imax]::Float64, # vx1
        evect[1,imin]::Float64, # vx2
        evect[2,imax]::Float64, # vy1
        evect[2,imin]::Float64, # vy2
        √(abs(eigval[imax]))::Float64, # a1 
        √(abs(eigval[imin]))::Float64, # a2
    )

end

function getFSE_healing(F, FSE)
    @batch for iel in eachindex(F)
        healing_FSE!(FSE, F, iel)
    end
    FSE, F
end 

function healing_FSE!(FSE, F, iel)
    
    @inbounds if FSE[iel].a1./FSE[iel].a2 > 1e3
        @show F[iel] = @SMatrix [1.0 0.0; 0.0 1.0]
        @show iel
    end

    # Compute FSE
    eigval, evect = eigen(F[iel] * F[iel]')
    if eigval[2] > eigval[1]
        imax, imin = 2, 1
    else
        imax, imin = 1, 2
    end

    # Fill FSE
    @inbounds FSE[iel] = FiniteStrainEllipsoid(
        evect[1,imax]::Float64, # vx1
        evect[1,imin]::Float64, # vx2
        evect[2,imax]::Float64, # vy1
        evect[2,imin]::Float64, # vy2
        √(abs(eigval[imax]))::Float64, # a1 
        √(abs(eigval[imin]))::Float64, # a2
    )

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
