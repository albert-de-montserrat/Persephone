# Rate of annealing
struct Annealing{T}
    rate::T # rate of annealing
end

*(a::Number, annealing::Annealing) = a*annealing.rate
*(annealing::Annealing, a::Number) = a*annealing.rate

#=
Unstrecth the FSE semi-axes by a factor of s.
The area of the ellipse must remain the same
after unstretching
=#
function unstretch_axes(a1, a2, s)
    # a2, a1 are the old semiaxes
    # s is the unstretching factor
    r0 = a2/a1 # old ratio 
    r = @muladd r0 + (1-r0)*s # new aspect ratio
    A = π*a2*a1 # Area of the FSE
    # find new unstrecthed axes such that the   
    # aspect ratio is r and A remains untouched
    # a1u = max(√(1/r), 1.0) # can't be lower than 1.0
    # a2u = min(r*a1u, 1.0) # can't be larger than 1.0
    a1u = √(A/(π*r))
    a2u = r*a1u

    return a1u, a2u
end

#=
Recover the new F tensor after unstretching the FSE semi-axes.
We know that F = LR, where L is the left-stretch tensor and R⁻¹=Rᵀ 
is the orthogonal rotation tensor. First we recover the new L using
the old eigenvalues (we do not want to rotate the FSE, only unstretch 
it) and new eigenvalues:
    L = PDP⁻¹
where P = [λ₁ λ₃ λ₃] and D = I⋅[a₁ a₃ a₃]ᵀ. R is obtained using the 
old F and L0 tensors:
    R = F\L0
and finally we obtain the new deformation gradient tensor:
    Fn = LR
=#
function recover_F(F, L0, λ11, λ12, λ21, λ22, a1, a2)
    P = @SMatrix [
        λ11 λ21
        λ12 λ22
    ]
    D = @SMatrix [
        a1*a1  0.0
        0.0    a2*a2
    ]
    # recover left-stretch tensor 
    L = P*D\P
    # rotation tensor R = F\(F*Fᵀ) = F\l0
    R = F\L0
    # unstretched tensor of deformation
    Fu = L*R
end