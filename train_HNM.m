function train_HNM(varargin)

    caffe.reset_all();
    caffe.set_mode_gpu();
    caffe.set_device(0);
    caffe_solver = caffe.Solver('./model/solver.prototxt');
    caffe_solver.net.copy_from('./model/50000.caffemodel');
    load('/data/zhangminwen/unet/data/pos_imgs.mat');
    load('/data/zhangminwen/unet/data/pos_masks.mat');
    load('/data/zhangminwen/unet/data/neg_imgs.mat');
    load('/data/zhangminwen/unet/data/neg_masks.mat');
    train_imgs = cat(4, neg_imgs, pos_imgs);
    train_masks = cat(4, neg_masks, pos_masks);
    neg_num = size(neg_imgs, 4);
    pos_num = size(pos_imgs, 4);
    neg_index = randperm(neg_num);
    while(neg_num>pos_num)
        
        shuffled_inds = [];
        iter_ = caffe_solver.iter();
        max_iter = 60000;
        neg_num_to_train = neg_index(1:3*pos_num);
        neg_index = neg_index(~ismember(neg_index, neg_num_to_train));
        train_index = [neg_num_to_train, (1:pos_num)+neg_num];
        while (iter_ < max_iter)
            [shuffled_inds, sub_inds] = generate_random_minibatch(shuffled_inds, 32, train_index);
            net_inputs = generate_input(sub_inds, train_imgs, train_masks);
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
    
end



function [shuffled_inds, sub_inds] = generate_random_minibatch(shuffled_inds, ims_per_batch, pos_num)
    if length(shuffled_inds)<pos_num
        %images_dir = dir('./data/train/images/*.tif');  
       shuffled_inds = pos_num;
    end
    sub_inds = shuffled_inds(1:ims_per_batch);
    shuffled_inds(1:ims_per_batch) = [];
end

function net_inputs = generate_input(sub_inds, train_imgs, train_masks)
    load('./data/mean_image.mat');
    ims = [];
    masks = [];
    for i = 1:length(sub_inds)
        %im = imread(['./data/train/images/', sprintf('%d.tif', sub_inds(i))]);
        im = train_imgs(:, :, :, sub_inds(i));
        im = single(im)-single(img_mean);
        %mask = imread(['./data/train/masks/', sprintf('%d.tif', sub_inds(i))]);
        mask = train_masks(:, :, :, sub_inds(i));
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
