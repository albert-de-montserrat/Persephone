# Out-of-place C = A*B
@inline function gemm(A::AbstractMatrix, B::AbstractMatrix)
    m, n, p = size(A,1), size(B,2), size(A,2)

    C = Matrix{eltype(A)}(undef, m, n)

    @turbo for i in 1:m, k in 1:n
        aux = zero(eltype(A[1]))
        for j in 1:p
            aux += A[i,j]*B[j,k]
        end
        C[i,k] = aux
    end
    return C

end 

# threaded version of vectorized gemm
@inline function gemmt(A::AbstractMatrix, B::AbstractMatrix)
    m, n, p = size(A,1), size(B,2), size(A,2)

    C = Matrix{eltype(A)}(undef, m, n)

    @tturbo for i in 1:m, k in 1:n
        aux = zero(eltype(A[1]))
        for j in 1:p
            aux += A[i,j]*B[j,k]
        end
        C[i,k] = aux
    end
    return C

end 

# In-place C = A*B
@inline function gemm!(C::AbstractArray{T},A::AbstractArray{T},B::AbstractArray{T}) where T
    
    m, n, p = size(A,1), size(B,2), size(A,2)
    @turbo for i in 1:m, k in 1:n
        aux = zero(eltype(A[1]))
        for j in 1:p
            aux += A[i,j]*B[j,k]
        end
        C[i,k] = aux
    end
end 

@inline function gemmt!(C::AbstractArray{T},A::AbstractArray{T},B::AbstractArray{T}) where T
    
    m, n, p = size(A,1), size(B,2), size(A,2)
    @tturbo for i in 1:m, k in 1:n
        aux = zero(eltype(A[1]))
        for j in 1:p
            aux += A[i,j]*B[j,k]
        end
        C[i,k] = aux
    end
end 

# x + p*y -> y (vec+scalar*vec)
function xpy!(y::Vector{T}, x::Vector{T}, p::T) where T
    @tturbo for i in eachindex(y)
        y[i] = x[i] + p * y[i]
    end
end

# out-of-place Sparse CSC matrix times dense vector
function spmv(A::AbstractSparseMatrix, x::DenseVector)
    out = zeros(eltype(x), A.m)
    Aj = rowvals(A)
    nzA = nonzeros(A)
    @inbounds for col in 1:A.n
        xj = x[col]
        @fastmath for j in nzrange(A, col)
            out[Aj[j]] += nzA[j]*xj
        end
    end
    out
end

# in-of-place Sparse CSC matrix times dense vector
function spmv!(A::AbstractSparseMatrix, x::DenseVector, out::DenseVector)
    fill!(out, zero(eltype(out)))
    Aj = rowvals(A)
    nzA = nonzeros(A)
    @inbounds for col in 1:A.n
        xj = x[col]
        @fastmath for j in nzrange(A, col)
            out[Aj[j]] += nzA[j]*xj
        end
    end
    out
end

mynorm(r::Vector) = sqrt(mydot(r,r))

# Dot product
function mydot(a::AbstractArray{T}, b::AbstractArray{T}) where {T}
    n = zero(T)
    @turbo for i in eachindex(a)
        n += a[i]*b[i]
    end
    n
end

# Threaded dot product
function mydott(a::AbstractArray{T}, b::AbstractArray{T}) where {T}
    n = zero(T)
    @tturbo for i in eachindex(a)
        n += a[i]*b[i]
    end
    n
end
