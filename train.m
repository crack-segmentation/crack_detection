
function train_unet(varargin)
    caffe.reset_all();
    caffe.set_mode_gpu();
    caffe.set_device(1);
    caffe_solver = caffe.Solver('./model/solver.prototxt');
    caffe_solver.net.copy_from('./model/20000.caffemodel');
    shuffled_inds = [];
    max_iter =  20000;
    iter_ = 0;
    while (iter_ < max_iter)
        [shuffled_inds, sub_inds] = generate_random_minibatch(shuffled_inds, 32);
        net_inputs = generate_input(sub_inds);
        th = tic();
        caffe_net_reshape_as_input(caffe_solver.net, net_inputs);
        caffe_net_set_input_data(caffe_solver.net, net_inputs);
        caffe_solver.step(1);
        t_proposal = toc(th);
        fprintf('%.3fs\n', t_proposal);
        
        if mod(iter_, 10000)==0 && iter_>1;
            caffe_solver.net.save(['./model/', num2str(iter_), '.caffemodel'])
        end
        iter_ = caffe_solver.iter();
        
    end
    
end


function [shuffled_inds, sub_inds] = generate_random_minibatch(shuffled_inds, ims_per_batch)
    if length(shuffled_inds)<ims_per_batch
       shuffled_inds = randperm(length(dir('./data/train/arg_pos_img/*.tif')));
    end
    sub_inds = shuffled_inds(1:ims_per_batch);
    shuffled_inds(1:ims_per_batch) = [];
end

function net_inputs = generate_input(sub_inds)
    load('./data/matfile2.mat');
    load('./data/matfile3.mat');
    ims = [];
    masks = [];
    for i = 1:length(sub_inds)
        im = imread(['./data/train/arg_pos_img/', sprintf('%d.tif', sub_inds(i))]);
        im = single(im)-single(mean_image_pos);
%         im_m  =  im./max(1e-8,single(im_nm));
%         im = single(im)-single(mean_image_pos);
        mask = imread(['./data/train/arg_pos_mask/', sprintf('%d.tif', sub_inds(i))]);
        mask = single(mask)/255;
        ims = cat(4, ims, im);
        masks = cat(4, masks, mask);
    end
    im_blob = ims(:, :, :, :);
    im_blob = permute(im_blob, [2, 1, 3, 4]);
    im_blob = single(im_blob);
    
    mask_blob = masks(:, :, :, :);
    mask_blob = permute(mask_blob, [2, 1, 3, 4]);
    mask_blob = single(mask_blob);
    net_inputs = {im_blob, mask_blob};
end
