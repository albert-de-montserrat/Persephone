function StokesPcCG_cuda(U,P,KK,MM,GG,Rhs,ifree)
    
    rtol_Pat = 1e-8
    itmax_Pat = 200
    itmin_Pat = 10
    itnum = 0

    # allocate some stuff
    Udev = CuVector(U)
    Pdev = CuVector(P)
    nU = length(Rhs) 
    z = CUDA.fill(0.0, nU)

    # forces resulting from pressure gradients
    GGdev = CuSparseMatrixCSC(GG)
    Rhsdev = CuVector(Rhs)
    Rhsdev .+= GGdev * Pdev
    Rhsdev_ifree = Rhsdev[ifree]

    # guess of the velocity field 
    KKdev = CuSparseMatrixCSR(
        KK[ifree, ifree]
    )
    sol, = Krylov.cg(KKdev, Rhsdev_ifree)
    Udev[ifree] .= sol

    Utemp = Udev[ifree]
    CUSOLVER.csrlsvchol!(KKdev, Rhsdev_ifree, Utemp)

    # initial pressure residual vector
    GGtransp_minus =  CuSparseMatrixCSC(sparse(-GG'))
    GGtransp =  CuSparseMatrixCSC(sparse(GG'))
    r = GGtransp_minus *Udev
    rm0space!(r)
    Prms = norm(r) # norm of r_i
    tol = Prms * rtol_Pat
    # get preconditioner
    # MMdev, pc = _preconditioner_cuda("jacobi", MM)
    _, pc = _preconditioner_cuda("lumped", MM)
    
    # Begin of Patera pressure iterations 
    # d = _precondition(pc, MM, r) # precondition residual
    d = _precondition(pc, r) # precondition residual
    q = deepcopy(d)  # define FIRST search direction q
    rlast = similar(r)

    to2 = TimerOutput()
    @timeit to2 "loop" for itPat = 1:itmax_Pat
        itnum +=1

        rm0space!(q)
        rd = dot(r,d) # numerator for alpha
        
        #= Perform the S times q multiplication ======================    
            S cannot be calculated explicitly, since Kinv cannot be formed
            Sq = S * q = (G' * Kinv * G) * q
            Hence, the muliplicatipon is done in 3 steps:
            (1) y   = G*q
            (2) K z = y (TODO PARDISO direct solver)
            (3) Sq  = G'*z
        =============================================================#
        y = GGdev*q
       
        @timeit to2 "cg" sol, = Krylov.cg(
            KKdev, 
            y[ifree]
        ) 
        z[ifree] .= sol
        
        Sq = GGtransp*z
        qSq = dot(q,Sq) # denominator to calculate alpha
        copyto!(rlast, r) # needed for Polak-Ribiere version 1
        α = rd/qSq # steps size in direction q
        
        # Update solution and residual  
        _updatesolution_cuda!(Pdev,Udev,r,q,z,Sq,α)
        # remove nullspace
        rm0space!(r)
        # Check convergence 
        Prrms = norm(r)
        if Prrms < tol && itPat>=itmin_Pat
            println("\n", itnum," CG iterations\n")
            break
        end

        d = _precondition(pc, r) # precondition rsidual
        # Make new search direction q S-orthogonal to all previous q's
        β  = dot(r-rlast,d)/rd # Polak-Ribiere version 1        
        @. q  = d + β*q # calculate NEW search direction
        # xpy!(q, d, β)
        
    end

    # _MKLrelease!(ps)

    # return Array(Udev), Array(Pdev)
    to2
end

# Function takes out the mean of vector a
# (thereby removes the constant pressure mode, i.e. the nullspace)
function rm0space!(r::CuArray)
    r .-= mean(r)
end 

@inline function _preconditioner_cuda(type, MM)
    if type == "diagonal"
        #=
            Inverse of the diagonal of the mass matrix scaled by viscosity
        =#
        MM, C = diagonalpreconditioner_cuda(MM)

    elseif type == "lumped"
        #=
            Inverse of lumped mass matrix scaled by viscosity
        =#        
        MM, C  = lumpedpreconditioner_cuda(MM)

    elseif type == "jacobi"
        #=
            Inverse of lumped mass matrix scaled by viscosity
        =#        
        MM, C  = lumpedpreconditioner_cuda(MM)
    end
    return MM, C
end

@inline function diagonalpreconditioner_cuda(MM)
    MM = MM .+ tril(MM,-1)'
    return CuSparseMatrixCSC(MM), CuSparseMatrixCSC(inv(diagm(MM)))
end  

@inline function lumpedpreconditioner_cuda(MM)
    MM = MM .+ tril(MM,-1)'
    return CuSparseMatrixCSC(MM), CuVector(dropdims(sum(MM,dims=2), dims=2).^-1)
end  

# PRECONDITIONING: lumped
@inline _precondition(pc::CuVector,r::CuVector) = r .* pc

@inline function _updatesolution_cuda!(P,U,r,q,z,Sq,α)
    @. P += α * q
    @. U += α * z
    @. r -= α * Sq
end
