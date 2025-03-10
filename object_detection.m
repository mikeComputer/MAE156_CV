function process_all_files()
    % List of .mat files
    file_paths = {
        'trial2.1_freq_0.1_update.mat', 
        'trial2.2_freq_0.1_update.mat', 
        'trial2.3_freq_0.1_update.mat', 
        'trial2.4_freq_0.1_update.mat', 
        'trial2.5_freq_0.1_update.mat', 
        'trial2.6_freq_0.1_update.mat'
    };
    
    % Process all files recursively
    process_files_recursive(file_paths, 1, 2);
end

function process_files_recursive(file_list, index, n)
    if index > length(file_list)
        return;
    end
    process_mat_file(file_list{index}, n);
    process_files_recursive(file_list, index + 1, n);
end

function process_mat_file(file_path, n)
    % Load the .mat file
    data = load(file_path);

    % Extract relevant variables
    t_received = data.t_received(:); % Ensure column vector
    cart2pos = data.cart2pos(:);     % Ensure column vector

    % Find peaks
    [pks, peakIndices] = findpeaks(cart2pos);

    % Extract peak times and values
    peakTimes = t_received(peakIndices);
    peakValues = pks;

    % Compute distances between consecutive peaks
    peakDistances = diff(peakTimes);

    % Find the maximum peak
    [y_0, maxIndex] = max(peakValues);
    t_0 = peakTimes(maxIndex);

    % Find the nth peak after the maximum peak
    if maxIndex + n <= length(peakValues)
        t_n = peakTimes(maxIndex + n);
        y_n = peakValues(maxIndex + n);
    else
        t_n = NaN;
        y_n = NaN;
    end

    % Find the first converging peak (y_inf)
    [t_inf, y_inf] = find_convergence_peak(peakTimes, peakValues, 10);

    % Display results
    fprintf('\nProcessing: %s\n', file_path);
    fprintf('n: %d\n', n);
    fprintf('t_n: %.3f\n', t_n);
    fprintf('y_n: %.3f\n', y_n);
    fprintf('t_0: %.3f\n', t_0);
    fprintf('y_0: %.3f\n', y_0);
    fprintf('t_inf: %.3f\n', t_inf);
    fprintf('y_inf: %.3f\n', y_inf);

    % Plot the signal with detected peaks
    figure;
    plot(t_received, cart2pos, 'b'); hold on;
    plot(peakTimes, peakValues, 'ro', 'MarkerFaceColor', 'r'); % Mark peaks
    xlabel('Time (s)');
    ylabel('Signal Amplitude');
    title(sprintf('Converging Sinusoidal Signal (cart2pos) - %s', file_path), 'Interpreter', 'none');
    legend('Signal', 'Peaks');
    grid on;
    hold off;
end

function [t_inf, y_inf] = find_convergence_peak(peakTimes, peakValues, threshold)
    % Identify the first peak when the signal converges
    for i = 1:length(peakValues)-1
        if abs(peakValues(i+1) - peakValues(i)) < threshold
            t_inf = peakTimes(i);
            y_inf = peakValues(i);
            return;
        end
    end
    t_inf = NaN;
    y_inf = NaN;
end
