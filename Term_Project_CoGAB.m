% OHM Term Project 
% Amalgamation of GA and ABC
% 19IM10039 - Debraj Chatterjee
% ***************************************************%
%% 
clc;
clear;
close all;

%% Input Data
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
max_gen = 40; %Max number of generations

CP = 0.8; %Crossover Probability
MP = 0.006;  %Mutation Probability
RR = 0.194;  %Reproduction Ratio
%% ABC settings
n_employed = n; %Number of employed bees
n_onlooker = n; %Number of onlooker bees
n_scout = round(2*n/20); %Number of scout bees (10% of whole swarm - emplyed+onlooker)
prob_setting = 0.1; % range till modification happens by onlooker bees

%% Binary 2 Decimal settings
max_bin = int2str(ones(1,c)); %Max binary value - '1111111111' in this case (All loans sanctioned)
min_bin = int2str(zeros(1,c));%Min binary value - '0000000000' in this case - (No loans sanctioned)
max_bin = max_bin(find(~isspace(max_bin))); %Removing spaces from string to bring to proper format
min_bin = min_bin(find(~isspace(min_bin)));

max_dec = bin2dec(max_bin); %Decimal value of max binary
min_dec = bin2dec(min_bin); %Decimal value of min binary

%% Random Population Initialization
tic; %Start stopwatch
population = randi([0,1],n,c); %Setting 0 or 1 randomly for n solutions of c dimensions each
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

%% GA with ABC refining the initial population

    %ARTIFICIAL BEE COLONY ALGORITHM
    pop_fitnesses = fitness(population, r_L, r_T, L, Loss, T, Beta, n, K, D);
    %EMPLOYED BEES MODIFY SOLUTIONS
    population = employed_bee_mod(population, pop_fitnesses, n, c,  r_L, r_T, L, Loss, T, Beta, K, D, max_dec, min_dec);
    pop_fitnesses = fitness(population, r_L, r_T, L, Loss, T, Beta, n, K, D); %population fitness after employee bees mod

    %ONLOOKER BEES MODifY SOLUTIONS
    population = onlooker_bee_mod(population, pop_fitnesses, n, c, prob_setting, max_dec, min_dec);
    pop_fitnesses = fitness(population, r_L, r_T, L, Loss, T, Beta, n, K, D); %Population fitness after onlooker bees mod

    %SCOUT BEES MODIFY SOLUTIONS
    population = scout_bee_mod(population, pop_fitnesses, n_scout, c);
    pop_fitnesses = fitness(population, r_L, r_T, L, Loss, T, Beta, n, K, D); %Population fitness after scout bees mod

for iter = 1:max_gen %Global Loop / Stopping Criteria
    %SELECTION - RULETTE WHEEL BASED
    parents_indices = zeros(n,1);
    pop_fitnesses = fitness(population,r_L, r_T, L, Loss, T, Beta, n, K,D);
    parent_pop = zeros(n,c);
    for parent = 1:n
        selected_index = roulette_wheel(pop_fitnesses);
        parent_pop(parent,:) = population(selected_index,:);
    end

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
toc;

disp(global_best_solution);
disp(global_best_fitness);
disp(iter_avg_fitness(max_gen+1,1));

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
% FITNESS CALCULATION
function F = fitness(population, r_L, r_T, L, Loss, T, Beta, pop_size,K,D)   
F = zeros(pop_size,1);
    for i = 1:pop_size
       rev(i) = sum(population(i,:).*(r_L.*L - Loss)); %Revenue from sanctioned loans
       expense(i) = sum(population(i,:).*(r_T.*T));

       F(i) = max(0,rev(i) + expense(i) - Beta - sum(population(i,:).*Loss)...
           - penalty(population(i,:),L,K,D));
    end

end

%PENALTY FUNCTION
function p = penalty(chromosome,L,K,D)
    multiplier  = 0.5;
    p = 0;
    if (sum(chromosome.*L) > (1-K)*D)%If amount lent out is gretaer than max limit
        p = sum(chromosome.*L) - (1-K)*D;   %apply penalty
        p = multiplier*p;
    end
end

% SELECTION FUNCTION
function selected_parent_index = roulette_wheel(fitnesses)
    n = size(fitnesses,1);
    random_num = rand();
    total_fitness = sum(fitnesses);
    cum_fit = zeros(n,1);
    fitness_ratio = zeros(n,1);
    cum_fit(1,1) = fitnesses(1,1);
    fitness_ratio(1,1) = cum_fit(1,1)/total_fitness; %Weights of roulette wheel
    if random_num<=fitness_ratio(1,1)
        selected_parent_index = 1;
    else 
        for i = 2:n
            cum_fit(i,1) = cum_fit(i-1,1) + fitnesses(i,1);
            fitness_ratio(i,1) = cum_fit(i,1)/total_fitness;
            if fitness_ratio(i-1,1)<random_num && fitness_ratio(i,1)>=random_num
                selected_parent_index = i;
                break;
            end
        end
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
    bin_sol = zeros(pop_size,dim_size); %Binary string representation of population
    dec_sol = zeros(pop_size,1); %Decimal representation of population
    after_employed_pop = []; %Population after employed bees modify the solutions
    v = []; %modifications made by employed bees
    for i = 1:pop_size
        after_employed_pop = [after_employed_pop ; blanks(dim_size)]; %Creating proper character array
        v = [v ; blanks(dim_size)];
    end

    for i = 1:pop_size
        temp = int2str(population(i,:)); %Creating binary string
        temp = temp(find(~isspace(temp))); %Removing spaces from the binary string
        bin_sol(i,:) = bin2num_array(temp); %Converting binary string to numerical array via defined function
        dec_sol(i,1) = bin2dec(temp); % Corresponding decimal value of binary string
    end
    

    for i = 1:pop_size
        phi = rand();
        k = randi([1,pop_size]);
        
        while(k==i)
            k = randi([1,pop_size]);
        end

        modified = dec_sol(i,1) + (phi*(dec_sol(i,1) - dec_sol(k,1))); %Modification equation by employed bees
        if modified>max_dec %If solution going out of bounds
            modified = max_dec; %Bring back to range
        end
        if modified<min_dec %If solution going out of bounds
            modified = min_dec; %Bring back to range
        end

        v(i,:) = dec2bin(modified, dim_size); %Binary string value of modified decimal solution
    end

    v = bin_pop2num_pop(v); %Converting population of binary string solutions to numerical array solutions to easily calculate fitness
    v_fitness = fitness(v, r_L, r_T, L, Loss, T, Beta, pop_size, K, D); %Fitness of modified solution population
    for i = 1:pop_size
        if v_fitness(i,1)>fitnesses(i,1) %Comparison between modified and original solutions
            after_employed_pop(i,:) = v(i,:);
        else
            after_employed_pop(i,:) = population(i,:);
        end
    end
end

% ONLOOKER BEES MODIFICATION
function after_onlooker_pop = onlooker_bee_mod(population, fitnesses, pop_size, dim_size, prob_setting, max_dec, min_dec)
    bin_sol = zeros(pop_size,dim_size); %Binary string representation of population
    dec_sol = zeros(pop_size,1); %Decimal representation of population
    after_onlooker_pop = []; %Population after onlooker bees modify
    for i = 1:pop_size
        after_onlooker_pop = [after_onlooker_pop ; blanks(dim_size)]; %Proper character array representation
    end


    for i = 1:pop_size
        temp = int2str(population(i,:)); %Creating binary string from numerical array
        temp = temp(find(~isspace(temp))); %Remove spaces to put in proper binary string representation
        bin_sol(i,:) = bin2num_array(temp); %Convert binary string to numerical array
        dec_sol(i,1) = bin2dec(temp); %Decimal value of binary string
    end 

prob = zeros(pop_size,1); %Probability value of each solution
    total_fitness = sum(fitnesses);
    for i = 1:pop_size
        prob(i,1) = fitnesses(i,1)/total_fitness; %Pi = Fi/sum(Fi)
        if prob(i,1)<=prob_setting %Criteraia for modification by onlooker bees
            phi = rand();
            k = randi([1,pop_size]);
            while(k~=i)
                k = randi([1,pop_size]);
            end

            dec_sol(i,1) = dec_sol(i,1) + phi*(dec_sol(i,1) - dec_sol(k,1)); %Modification equation
            if dec_sol(i,1)>max_dec %If going out of range, bring back
            dec_sol(i,1) = max_dec;
            end
            if dec_sol(i,1)<min_dec %If going out of range bring back
            dec_sol(i,1) = min_dec;
            end

        end
    end

    for i = 1:pop_size
        after_onlooker_pop(i,:) = dec2bin(dec_sol(i,1), dim_size); %Creating binary string from decimal value
    end
    after_onlooker_pop = bin_pop2num_pop(after_onlooker_pop); %Creating numerical array population from binary population
    %for easy calculation of fitness
end

%SCOUT BEES MODIFICATION FUNCTION
function after_scout_pop = scout_bee_mod(population, fitnesses, n_scout, dim_size)
    after_scout_pop = population;
    for scout_bee = 1:n_scout
        [~,min_idx] = min(fitnesses); %recording index of solution with lowest fitness
        fitnesses(min_idx,1) = 9999; %Changing the fitness to a high value so that it is not selected again
        after_scout_pop(min_idx,:) = randi([0,1], 1, dim_size); %Worst solutions replaced by randomly generated ones - EXPLORATION
    end
end

%FUNCTION TO CONVERT BINARY POPULATION TO NUMERIC ARRAY POPULATION
%NUMERICA ARRAY REPRESENTATION IS REQUIRED AS FITNESS FUNCTION EVALUATES IN
%THAT FORM
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
            num_array(1,pos) = ' '; %Adding spaces between numbers because then str2num function properly separates as numbers in different cells
        else
            num_array(1,pos) = binary_string(1,(pos+1)/2); %Numbers from the binary string in alternate positions
        end
    end
    num_array = str2num(num_array); %Converting binary string with spaces to required numerical array
end




