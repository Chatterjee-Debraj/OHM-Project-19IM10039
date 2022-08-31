% OHM Term Project 
% Amalgamation of GA and ABC
% 19IM10039 - Debraj Chatterjee
% ***************************************************%
%Will try to replace selection of GA with ABC
%% %% Input Data
D = 60; %Financial Institution Deposit
K = 0.15;  %Required Reserve Ratio
L = [10 25 4 11 18 3 17 15 9 10]; %Loan Sizes
r_L = [0.021 0.022 0.021 0.027 0.025 0.026 0.023 0.021 0.028 0.022];%Interest Rates
Loss = [0.0002 0.0058 0.0001 0.0003 0.0024 0.0002 0.0058 0.0002 0.0001 0.0001]; %Expected Loss
Rating = ['AAA' 'BB' 'A' 'AA' 'BBB' 'AAA' 'BB' 'AAA' 'A' 'A'];
r_T = 0.01; %Customer transaction Rate
r_D = 0.009; %Deposit Rate

Delta = 0.0025; %Institutional cost
Beta = r_D*D;  %Cost of Demand Deposit
T = (1-K)*D - L;  %Institutional Transaction Cost
Loan_Cost = L*Delta; %Loan cost of each loan
Total_Trans_Cost = r_T.*T;

Revenue = r_L.*L - Loss; %Expected Revenues from each loan if sanctioned
c = numel(L); %Number of customers
%% GA settings
n = 60; %Population Size
max_gen = 30; %Max number of generations

CP = 0.8; %Crossover Probability
MP = 0.006;  %Mutation Probability
RR = 0.194;  %Reproduction Ratio
%% ABC settings
n_employed = n; %Number of employed bees
n_onlooker = n; %Number of onlooker bees
n_scout = round(2*n/10); %Number of scout bees (10% of whole swarm - emplyed+onlooker)
prob_setting = 0.1;

%% Binary 2 Decimal settings
max_bin = int2str(ones(1,c));
min_bin = int2str(zeros(1,c));
max_bin = max_bin(find(~isspace(max_bin)));
min_bin = min_bin(find(~isspace(min_bin)));

max_dec = bin2dec(max_bin);
min_dec = bin2dec(min_bin);




%% Random Population Initialization
population = randi([0,1],n,c);
global_best_fitness = 0;
global_best_solution = zeros(1,c);
iter_best_fitness = zeros(max_gen+1,1);
iter_avg_fitness = zeros(max_gen+1,1);
iter_best_index = zeros(max_gen+1,1);

pop_fitnesses = fitness(population, r_L, r_T, L, Loss, T, Beta, n, K, D);
[iter_best_fitness(1,1), iter_best_index(1,1)] = max(fitness(population,r_L, r_T, L, Loss, T, Beta, n, K, D));
global_best_fitness = iter_best_fitness(1,1);
iter_avg_fitness(1,1) = mean(fitness(population, r_L, r_T, L, Loss, T, Beta, n, K, D));
global_best_solution = population(iter_best_index(1,1),:);

%% GA with ABC for Selection
for iter = 1:max_gen
    
    %ABC
    pop_fitnesses = fitness(population, r_L, r_T, L, Loss, T, Beta, n, K, D);
    %EMPLOYED BEES
    population = employed_bee_mod(population, pop_fitnesses, n, c,  r_L, r_T, L, Loss, T, Beta, K, D, max_dec, min_dec);
    pop_fitnesses = fitness(population, r_L, r_T, L, Loss, T, Beta, n, K, D);

    %ONLOOKER BEES
    population = onlooker_bee_mod(population, pop_fitnesses, n, c, prob_setting, max_dec, min_dec);
    pop_fitnesses = fitness(population, r_L, r_T, L, Loss, T, Beta, n, K, D);

    %SCOUT BEES
    population = scout_bee_mod(population, pop_fitnesses, n_scout, c);
    pop_fitnesses = fitness(population, r_L, r_T, L, Loss, T, Beta, n, K, D);

    %SELECTION - RANDOM
    selected_parents_index = randi([1,n],1,n);
    parent_pop = zeros(n,c);
    parent_pop = population(selected_parents_index,:);

     %CROSSOVER
    intermediate_pop = parent_pop;
    for parent_set = 1:n/2
        parent1(1,:) = parent_pop(parent_set*2 - 1,:);
        parent2(1,:) = parent_pop(parent_set*2,:);

        crossover_random = rand();
        if crossover_random<=CP
            two_children = two_point_crossover(parent1,parent2);
            intermediate_pop(parent_set*2-1,:) = two_children(1,:);
            intermediate_pop(parent_set*2,:)  = two_children(2,:);
        else 
            intermediate_pop(parent_set*2 - 1,:) = parent1(1,:);
            intermediate_pop(parent_set*2,:) = parent2(1,:);
        end


    end

    %MUTATION
    for individual = 1:n
        for gene = 1:c
            mutation_num = rand();
            if mutation_num<=MP
                intermediate_pop(individual,gene) = ...
                digit_flip(intermediate_pop(individual,gene));
            end
        end
    end

    %FITNESS COMPARISON
    intermediate_pop_fitness = fitness(intermediate_pop,r_L, r_T, L, Loss, T, Beta, n, K,D);
    parent_pop_fitness = fitness(parent_pop, r_L, r_T, L, Loss, T, Beta, n, K,D);
    for i = 1:n
        if intermediate_pop_fitness < parent_pop_fitness
        intermediate_pop(i,:) = parent_pop(i,:);
        end
    end
    population = intermediate_pop;
    [iter_best_fitness(iter+1,1),iter_best_index] = max(fitness(population,r_L, r_T, L, Loss, T, Beta, n, K,D));
    iter_avg_fitness(iter+1,1) = mean(fitness(population,r_L, r_T, L, Loss, T, Beta, n, K,D));

    if iter_best_fitness(iter+1,1)>global_best_fitness
        global_best_fitness = iter_best_fitness(iter+1,1);
        global_best_solution = population(iter_best_index,:);
    end
end

disp(global_best_solution);
disp(global_best_fitness);

%% 
subplot(1, 2, 1);
plot(1:max_gen+1, iter_best_fitness);
xlabel('Iterations');
ylabel('Best Fitness of population');
subplot(1, 2, 2);
plot(1:max_gen+1, iter_avg_fitness);
xlabel('Iterations');
ylabel('Average Fitness of population');

%% Helping Functions
% Fitness Calculation
function F = fitness(population, r_L, r_T, L, Loss, T, Beta, pop_size,K,D)   
F = zeros(pop_size,1);
    for i = 1:pop_size
        rev(i) = sum(population(i,:).*(r_L.*L - Loss)); %Revenue from sanctioned loans
        expense(i) = sum(population(i,:).*(r_T.*T));

       % F(i) = rev(i)-expense(i)-Beta-sum(population(i,:).*Loss) - ...
               %penalty(population(i,:),L,K,D);
       F(i) = max(0,rev(i) + expense(i) - Beta - sum(population(i,:).*Loss)...
           - penalty(population(i,:),L,K,D));
    end

end

%penalty function
function p = penalty(chromosome,L,K,D)
    multiplier  = 0.5;
    p = 0;
    if (sum(chromosome.*L) > (1-K)*D)
        p = sum(chromosome.*L) - (1-K)*D;
        p = multiplier*p;
    end
end

%TWO-POINT CROSSOVER FUNCTION
function two_children = two_point_crossover(parent1, parent2)
    c = size(parent1,2);
    parent1 = parent1;
    parent2 = parent2;
    child1 = parent1;
    child2 = parent2;
    point1 = randi(c);
    point2 = randi(c);

    for point = point1+1 : point2
        child1(1,point) = parent2(1,point);
        child2(1,point) = parent1(1,point);
    end
    two_children = [child1;
                    child2];
end

%MUTATION HELPER FUNCTION
function flipped_digit = digit_flip(digit)
        if digit == 0
            flipped_digit = 1;
        else
            flipped_digit = 0;
        end
    end

%EMPLOYED BEE MODIFICATION
function after_employed_pop = employed_bee_mod(population, fitnesses, pop_size, dim_size, r_L, r_T, L, Loss, T, Beta, K, D, max_dec, min_dec)    
    bin_sol = zeros(pop_size,dim_size);
    dec_sol = zeros(pop_size,1);
    after_employed_pop = [];
    v = [];
    for i = 1:pop_size
        after_employed_pop = [after_employed_pop ; blanks(dim_size)];
        v = [v ; blanks(dim_size)];
    end

    for i = 1:pop_size
        temp = int2str(population(i,:));
        temp = temp(find(~isspace(temp)));
        bin_sol(i,:) = bin2num_array(temp);
        dec_sol(i,1) = bin2dec(temp);
    end
    

    for i = 1:pop_size
        phi = rand();
        k = randi([1,pop_size]);
        
        while(k==i)
            k = randi([1,pop_size]);
        end

        modified = dec_sol(i,1) + (phi*(dec_sol(i,1) - dec_sol(k,1)));
        if modified>max_dec
            modified = max_dec;
        end
        if modified<min_dec
            modified = min_dec;
        end

        v(i,:) = dec2bin(modified, dim_size);
    end

    v = bin_pop2num_pop(v);
    v_fitness = fitness(v, r_L, r_T, L, Loss, T, Beta, pop_size, K, D);
    for i = 1:pop_size
        if v_fitness(i,1)>fitnesses(i,1)
            after_employed_pop(i,:) = v(i,:);
        else
            after_employed_pop(i,:) = population(i,:);
        end
    end
    %after_employed_pop = bin_pop2num_pop(after_employed_pop);
end

% ONLOOKER BEES MODIFICATION
function after_onlooker_pop = onlooker_bee_mod(population, fitnesses, pop_size, dim_size, prob_setting, max_dec, min_dec)
    bin_sol = zeros(pop_size,dim_size);
    dec_sol = zeros(pop_size,1);
    after_onlooker_pop = [];
    for i = 1:pop_size
        after_onlooker_pop = [after_onlooker_pop ; blanks(dim_size)];
    end


    for i = 1:pop_size
        temp = int2str(population(i,:));
        temp = temp(find(~isspace(temp)));
        bin_sol(i,:) = bin2num_array(temp);
        dec_sol(i,1) = bin2dec(temp);
    end 

prob = zeros(pop_size,1);
    total_fitness = sum(fitnesses);
    for i = 1:pop_size
        prob(i,1) = fitnesses(i,1)/total_fitness;
        if prob(i,1)<=prob_setting
            phi = rand();
            k = randi([1,pop_size]);
            while(k~=i)
                k = randi([1,pop_size]);
            end

            dec_sol(i,1) = dec_sol(i,1) + phi*(dec_sol(i,1) - dec_sol(k,1));
            if dec_sol(i,1)>max_dec
            dec_sol(i,1) = max_dec;
            end
            if dec_sol(i,1)<min_dec
            dec_sol(i,1) = min_dec;
            end

        end
    end

    for i = 1:pop_size
        after_onlooker_pop(i,:) = dec2bin(dec_sol(i,1), dim_size);
    end
    after_onlooker_pop = bin_pop2num_pop(after_onlooker_pop);
end

%SCOUT BEES MODIFICATION FUNCTION
function after_scout_pop = scout_bee_mod(population, fitnesses, n_scout, dim_size)
    after_scout_pop = population;
    for scout_bee = 1:n_scout
        [~,min_idx] = min(fitnesses);
        fitnesses(min_idx,1) = 9999;
        after_scout_pop(min_idx,:) = randi([0,1], 1, dim_size);
    end
end

%FUNCTION TO CONVERT BINARY POPULATION TO NUMERIC ARRAY
function num_pop = bin_pop2num_pop(binary_pop)
    pop_size = size(binary_pop,1);
    dim_size = size(binary_pop,2);
    num_pop = zeros(pop_size,dim_size);

    for i = 1:pop_size
        num_pop(i,:) = bin2num_array(binary_pop(i,:));
    end
end

%FUNCTION TO CONVERT BINARY STRING TO NUMERIC ARRAY
function num_array = bin2num_array(binary_string)
    bin_len = size(binary_string,2);
    num_array = blanks(2*bin_len - 1);
    
    for pos = 1:2*bin_len - 1
        if mod(pos,2)==0
            num_array(1,pos) = ' ';
        else
            num_array(1,pos) = binary_string(1,(pos+1)/2);
        end
    end
    %num_array = char(num_array);
    num_array = str2num(num_array);
end




