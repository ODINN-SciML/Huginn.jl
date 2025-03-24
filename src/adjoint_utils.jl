export compute_numerical_gradient, stats_err_backward, printVecScientific

function compute_numerical_gradient(x, args, fct, ϵ)
    grad = zero(x)
    grad_vec = vec(grad) # Points to the same position in memory
    x_ϵ = deepcopy(x)
    x_ϵ_vec = vec(x_ϵ)
    f0 = fct(x, args)
    @showprogress desc="Computing gradient with ϵ=$(ϵ)..." for i in range(1,length(x))
        x_ϵ .= x
        x_ϵ_vec[i] += ϵ
        grad_vec[i] = (fct(x_ϵ, args)-f0)/ϵ
    end
    return grad
end

function stats_err_backward(grad_analytical, grad_numeric)
    ratio = sqrt(sum(grad_analytical.^2))/sqrt(sum(grad_numeric.^2))-1
    angle = sum(grad_analytical.*grad_numeric)/(sqrt(sum(grad_analytical.^2))*sqrt(sum(grad_numeric.^2)))-1
    relerr = sqrt(sum((grad_analytical-grad_numeric).^2))/sqrt(sum((grad_analytical).^2))
    return ratio, angle, relerr
end

printVecScientific(v) = join([@sprintf("%9.2e", e) for e in v], " ")
