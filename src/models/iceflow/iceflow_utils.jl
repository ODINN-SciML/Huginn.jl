export inn1

# Helper functions for the staggered grid

####  Non-allocating functions  ####

"""
    diff_x!(O, I, Δx)

Compute the finite difference of array `I` along the first dimension and store the result in array `O`. The difference is computed using the spacing `Δx`.

# Arguments
- `O`: Output array to store the finite differences.
- `I`: Input array from which finite differences are computed.
- `Δx`: Spacing between points in the first dimension.

# Notes
- The function uses `@views` to avoid copying data when slicing arrays.
- The operation is performed in-place, modifying the contents of `O`.
"""
diff_x!(O, I, Δx) = @views @. O = (I[begin+1:end,:] - I[1:end-1,:]) / Δx

"""
    diff_y!(O, I, Δy)

Compute the finite difference along the y-axis and store the result in `O`.

# Arguments
- `O`: Output array where the result will be stored.
- `I`: Input array from which the finite difference is computed.
- `Δy`: The spacing between points in the y-direction.

# Description
This function calculates the finite difference along the y-axis for the input array `I` and stores the result in the output array `O`. The calculation is performed using the formula:

    O = (I[:,begin+1:end] - I[:,1:end - 1]) / Δy

The `@views` macro is used to avoid copying data when slicing the array.
"""
diff_y!(O, I, Δy) = @views @. O = (I[:,begin+1:end] - I[:,1:end - 1]) / Δy

"""
    avg!(O, I)

Compute the average of adjacent elements in the input array `I` and store the result in the output array `O`.

# Arguments
- `O`: Output array where the averaged values will be stored.
- `I`: Input array from which the adjacent elements will be averaged.

# Details
This function uses the `@views` macro to avoid creating temporary arrays and the `@.` macro to broadcast the operations. The averaging is performed by taking the mean of each 2x2 block of elements in `I` and storing the result in the corresponding element in `O`.
"""
avg!(O, I) = @views @. O = (I[1:end-1,1:end-1] + I[2:end,1:end-1] + I[1:end-1,2:end] + I[2:end,2:end]) * 0.25

"""
    avg_x!(O, I)

Compute the average of adjacent elements along the first dimension of array `I` and store the result in array `O`.

# Arguments
- `O`: Output array where the averaged values will be stored.
- `I`: Input array from which adjacent elements will be averaged.
"""
avg_x!(O, I) = @views @. O = (I[1:end-1,:] + I[2:end,:]) * 0.5

"""
    avg_y!(O, I)

Compute the average of adjacent elements along the second dimension of array `I` and store the result in array `O`.

# Arguments
- `O`: Output array where the averaged values will be stored.
- `I`: Input array from which the adjacent elements will be averaged.
"""
avg_y!(O, I) = @views @. O = (I[:,1:end-1] + I[:,2:end]) * 0.5

####  Allocating functions  ####

"""
    avg(A::AbstractArray)

Compute the average of each 2x2 block in the input array `A`. The result is an array where each element is the average of the corresponding 2x2 block in `A`.

# Arguments
- `A::AbstractArray`: A 2D array of numerical values.

# Returns
- A 2D array of the same type as `A`, where each element is the average of a 2x2 block from `A`.
"""
@views avg(A) = 0.25 .* ( A[1:end-1,1:end-1] .+ A[2:end,1:end-1] .+ A[1:end-1,2:end] .+ A[2:end,2:end] )

"""
    avg_x(A::AbstractArray)

Compute the average of adjacent elements along the first dimension of the array `A`.

# Arguments
- `A::AbstractArray`: Input array.

# Returns
- An array of the same type as `A` with one less element along the first dimension, containing the averages of adjacent elements.
"""
@views avg_x(A) = 0.5 .* ( A[1:end-1,:] .+ A[2:end,:] )

"""
    avg_y(A::AbstractArray)

Compute the average of adjacent elements along the second dimension of the input array `A`.

# Arguments
- `A::AbstractArray`: An array of numeric values.

# Returns
- An array of the same type as `A` containing the averages of adjacent elements along the second dimension.
"""
@views avg_y(A) = 0.5 .* ( A[:,1:end-1] .+ A[:,2:end] )

"""
    diff_x(A::AbstractArray)

Compute the difference along the first dimension of the array `A`.

# Arguments
- `A::AbstractArray`: Input array.

# Returns
- An array of the same type as `A` containing the differences along the first dimension.
"""
@views diff_x(A) = (A[begin + 1:end, :] .- A[1:end - 1, :])
diff_x(A, Δx) = diff_x(A) ./ Δx

"""
    diff_y(A::AbstractArray)

Compute the difference between adjacent elements along the second dimension (columns) of the input array `A`.

# Arguments
- `A::AbstractArray`: An array of numeric values.

# Returns
- An array of the same type as `A` containing the differences between adjacent elements along the second dimension.
"""
@views diff_y(A) = (A[:, begin + 1:end] .- A[:, 1:end - 1])
diff_y(A, Δy) = diff_y(A) ./ Δy

"""
    inn(A::AbstractArray)

Extracts the inner part of a 2D array `A`, excluding the first and last rows and columns.

# Arguments
- `A::AbstractArray`: A 2D array from which the inner part will be extracted.

# Returns
- A subarray of `A` containing all elements except the first and last rows and columns.
"""
@views inn(A) = A[2:end-1,2:end-1]

"""
    inn1(A::AbstractArray)

Returns a view of the input array `A` excluding the last row and the last column.

# Arguments
- `A::AbstractArray`: The input array from which a subarray view is created.

# Returns
- A view of the input array `A` that includes all elements except the last row and the last column.
"""
@views inn1(A) = A[1:end-1,1:end-1]

"""
    d2dx(f::Matrix{T}, i::Int, j::Int, Δx::Float64) where T <: Real

Compute the second central difference in the x-direction at (i,j).
"""
function d2dx(f::Matrix{T}, i::Int, j::Int, Δx::Float64) where T <: Real
    return (f[i+1, j] - 2f[i, j] + f[i-1, j]) / (Δx^2)
end

"""
    d2dy(f::Matrix{T}, i::Int, j::Int, Δy::Float64) where T <: Real

Compute the second central difference in the y-direction at (i,j).
"""
function d2dy(f::Matrix{T}, i::Int, j::Int, Δy::Float64) where T <: Real
    return (f[i, j+1] - 2f[i, j] + f[i, j-1]) / (Δy^2)  
end

"""
    d2dxy(f::Matrix{T}, i::Int, j::Int, Δx::Float64, Δy::Float64) where T <: Real

Compute the mixed second central difference (∂²f/∂x∂y) at (i,j).
"""
function d2dxy(f::Matrix{T}, i::Int, j::Int, Δx::Float64, Δy::Float64) where T <: Real
    return (f[i+1, j+1] - f[i+1, j-1] - f[i-1, j+1] + f[i-1, j-1]) / (4 * Δx * Δy)
end

"""
    project_curvatures(H, eₚ, eₛ)

Computes the scalar second derivative of the surface in a specific direction.

# Arguments
- `H`: Hessian matrix (2x2).
- `eₚ`: Principal direction vector 1 (2x1).
- `eₛ`: Principal direction vector 2 (2x1). 
# Returns
- `Kₚ`: Curvature in the direction of `eₚ`.
- `Kₛ`: Curvature in the direction of `eₛ`.
"""
function project_curvatures(H, eₚ, eₛ)
    Kₚ = eₚ' * H * eₚ
    Kₛ = eₛ' * H * eₛ
    return Kₚ, Kₛ
end

include("SIA2D/SIA2D_utils.jl")

"""
"""
function ∇slope(S::Matrix{T}, Δx::T, Δy::T) where T <: Real
    dSdx = diff_x(S) / Δx
    dSdy = diff_y(S) / Δy
    ∇Sx = avg_y(dSdx)
    ∇Sy = avg_x(dSdy)
    return (∇Sx.^2 .+ ∇Sy.^2).^(1/2)
end