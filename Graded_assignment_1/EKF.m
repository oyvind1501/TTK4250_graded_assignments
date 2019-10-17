classdef EKF
    % FILL IN THE DOTS
    properties
        model
        
        f % discrete prediction function
        F % jacobian of prediction function
        Q % additive discrete noise covariance
        
        h % measurement function
        H % measurement function jacobian
        R % additive measurement noise covariance
    end
    methods 
        function obj = EKF(model)
            obj = obj.setModel(model);
        end
        
        function obj = setModel(obj, model)
           % sets the internal functions from model
           obj.model = model;
           
           obj.f = model.f;
           obj.F = model.F;
           obj.Q = model.Q;
           
           obj.h = model.h;
           obj.H = model.H;
           obj.R = model.R;
        end
        
        function [xp, Pp] = predict(obj, x, P, Ts)
            % returns the predicted mean and covariance for a time step Ts
    
            Fk = obj.F(x, Ts);
            %Fk = cell2mat(Fk);
            xp = Fk*x;
            Qr = obj.Q(x, Ts);
            Pp = Fk*P*Fk' + Qr;
            %xp = [xp(1,1),xp(2,2),xp(3,3),xp(4,4),xp(5,5)];
            
            
        end
        function [vk, Sk] = innovation(obj, z, x, P)
            % returns the innovation and innovation covariance
            Hk = obj.H(x);
      
            
            z(1:2);
            
            vk = z(1:2) - Hk*x;
            %obj.R(1)
            Sk = Hk*P*Hk' + obj.R(1);
            
            
        end

        function [xupd, Pupd] = update(obj, z, x, P)
            % returns the mean and covariance after conditioning on the
            % measurement
            
            [vk, Sk] = obj.innovation(z, x, P);
            Hk = obj.H(x);
            Wk = P*(Hk'/Sk);
            xupd= x+Wk*vk;
            I =eye(size(P));
            Pupd=(I-Wk*Hk)*P*(I-Wk*Hk)' + Wk*feval(obj.R)*Wk';
            
            %Pupd=(I-Wk*Hk)*P*(I-Wk*Hk)'+Wk*obj.R*Wk';
            
            
            
%             [vk, Sk] = obj.innovation(z', x, P);
%             Hk = obj.H(x);
%             I = eye(size(P));
%             
%             Wk = P*Hk'*inv(Sk);
%             xupd = x + Wk*vk;
%             Pupd = (I - Wk*Hk)*P;
        end

        function NIS = NIS(obj, z, x, P)
            % returns the normalized innovation squared
            [vk, Sk] = obj.innovation(z, x, P);
            NIS = vk'*inv(Sk)*vk;
            
        end
        
        function NEES = NEES_VEL(obj,z,x,P)
            temps = zeros(2,1);
            NEES = 0;
            Z = zeros(2,size(z,2));
            for j = 1:size(z,2)
                temps(1:2) = Z(:,j);
                
                
                NEES = NEES + (temps - x)'*inv(P)*(temps - x);
                
            end
            NEES = NEES / size(z,2);
            
        end
        
        function NEES = NEES_POS(obj,z,x,P)
            temps = zeros(2,1);
            NEES = 0;
            
            for j = 1:size(z,2)
                temps(1:2) = z(1:2,j);
                NEES = NEES + (temps - x)'*inv(P)*(temps - x);
            end
            NEES = NEES / size(z,2);
            
        end

        
        function NEES = NEES(obj,z,x,P)
            temps = zeros(4,1);
            NEES = 0;
            
            for j = 1:size(z,2)
                temps(1:2) = z(1:2,j);
                NEES = NEES + (temps - x)'*inv(P)*(temps - x);
            end
            NEES = NEES / size(z,2);
            
        end

        function ll = loglikelihood(obj, z, x, P)
            % returns the logarithm of the marginal mesurement distribution
            [~, Sk] = obj.innovation(z, x, P);
    
            NIS = obj.NIS(z, x, P);
            ll = -0.5 * (NIS + log(det(2 * pi * Sk)));
 % FIKS
        end

    end
end

