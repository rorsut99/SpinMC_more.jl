using LinearAlgebra
using Plots


function iterate(state,stateNext,heff,lamb,omega,phi,dt,t,omegaz,phiz,lambz,hnoise)
    Sx=0.5*[0+0.0im 1.0+0im 
    1.0+0im 0+0im]
    Sy=0.5*[0 -1.0im
    1.0im 0]
    Sz=0.5*[1.0+0im 0
    0 -1.0+0im]

    Ham=(heff+hnoise[2])*Sy+lamb*sin(omega*t+phi)*Sx+lambz*cos(omegaz*t+phiz)*Sz+hnoise[1]*Sx+hnoise[3]*Sz

    stateMid=0.5*(state+stateNext)
    stateNew=state-1im*dt*Ham*stateMid

    return(stateNew)
end

function step(currentState,heff,lamb,omega,phi,dt,t,omegaz,phiz,lambz)
    nextGuess=deepcopy(currentState)
    hnoise=0*rand(Uniform(-0.05,0.05),3)

    for i in 1:10
        nextGuess=iterate(currentState,nextGuess,heff,lamb,omega,phi,dt,t,omegaz,phiz,lambz,hnoise)
    end

    currentState=deepcopy(nextGuess)
    return(currentState)
end

function expVal(state, Sx, Sy, Sz)
    vec = zeros(3)
    vec[1] = dot(state,Sx*state)
    vec[2] = dot(state,Sy*state)
    vec[3] = dot(state,Sz*state)

    return vec
end


function finalState(states,iters)
    Sx=0.5*[0+0.0im 1.0+0im 
    1.0+0im 0+0im]
    Sy=0.5*[0 -1.0im
    1.0im 0]
    Sz=0.5*[1.0+0im 0
    0 -1.0+0im]
    expValStates = zeros(3, iters)

    for i in 1:iters
        expValStates[:,i] = expVal(states[:,i], Sx, Sy, Sz)
    end

    return expValStates

end

function energy(state,heff,lamb,omega,phi,dt,t,omegaz,phiz,lambz)
    Sx = state[1]
    Sy = state[2]
    Sz = state[3]
    Ham=heff*Sy+lamb*sin(omega*t+phi)*Sx+lambz*cos(omegaz*t+phiz)*Sz

    return Ham



end
        





state=[1.0+0.0im,0.0-1im]
heff=-12.0
lamb=0.4
omega=1.0
phi=0.0
lambz=-0.025
omegaz=1.0
phiz=0.0
dt=0.01
t=0


nIter=5000
states=Matrix{ComplexF64}(undef,2,nIter)

Sx=0.5*[0+0.0im 1.0+0im 
    1.0+0im 0+0im]
Sy=0.5*[0 -1.0im
    1.0im 0]
Sz=0.5*[1.0+0im 0
    0 -1.0+0im]

# vars=rand(Uniform(-1.0,1.0),4)
# state=[vars[1]+vars[2]*im,vars[3]+vars[4]*im]
# state/=norm(state)



for i in 1:nIter
    
    states[:,i] = state
    global t=(i-1)*dt
    global state=step(state,heff,lamb,omega,phi,dt,t,omegaz,phiz,lambz)
end

expVals = finalState(states, nIter)

energySeries = zeros(nIter)
for i in 1:nIter
    energySeries[i] = energy(expVals[:,i],heff,lamb,omega,phi,dt,t,omegaz,phiz,lambz)
end



tpoints=zeros(nIter)

for i in 1:nIter
    tpoints[i]=dt*(i-1)
end