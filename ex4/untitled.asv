delta3_total = 0;
delta2_total = 0;

new_y = zeros(m,num_labels);
for i=1:num_labels
    new_y(:,i) = y == i;
end

t=1;
x = X(t,:)';
a1 = [1; x];
z2 = Theta1*a1;
a2 = sigmoid(z2);
a2 = [1;a2];
z3 = Theta2*a2;
h = sigmoid(z3);
delta3 = h - new_y(t,:)';
delta2 = Theta2'*delta3;
delta2 = delta2(2:end).*sigmoidGradient(z2);
delta3_total = delta3_total + delta3*a2';
delta2_total = delta2_total + delta2*a1';
size(delta2_total)
size(Theta1)