
function init_shadow() {


    const v = tf.tensor2d([[1, 2], [3, 4]]);
    const b = tf.tensor1d([1, 2]);

    const W = [tf.tensor2d([[1, 2], [3, 4]])];
    const U = [tf.tensor2d([[1, 2], [3, 4]])];
    const Ub = [tf.tensor1d([1, 2])];
    const Wb = [tf.tensor1d([1, 2])];

	function normalize(r, t) {
		size = r.length;
		return tf.tidy(() => {
		    r = tf.tensor2d(r, [1, size]);
		    t = tf.tensor2d(t, [1, size]);
		    m = tf.mean(r, axis=1, keepDims=true);
		    r = tf.div(r, m);
		    out = tf.concat([r, tf.mul(t, r)], 0)
			return out;
		});
	}

	function compute(input) {
		return tf.tidy(() => {
			function residue_layer(input, i) {
				const w = W[i];
				const u = U[i];
				const ub = Ub[i];
				const wb = Wb[i];

		        const residue = tf.mul(tf.elu(tf.add(tf.matMul(input, u), ub)), input)
		        const output = tf.add(tf.elu(tf.add(tf.matMul(residue, w), wb)), residue)
				return output;
			}

		    a = input;
		    for(var i = 0;i<W.length;++i)
		        a = residue_layer(a, i);

		    const output = tf.elu(tf.add(tf.matMul(a, v), b));
		    return output;
		});
	}


	function compare(f0, f1) {
		return tf.tidy(() => {
			return tf.exp(tf.sum(tf.squaredDifference(f0.expandDims(1), f1.expandDims(0)), 2).neg());
		});
	}

	function get_class(response) {
		return tf.tidy(() => {
			return tf.argMax(response, 1);
		});
	}

    function point_dist(p0, p1) {
        return Math.sqrt((p0[0] - p1[0])*(p0[0] - p1[0]) + (p0[1] - p1[1])*(p0[1] - p1[1]));
    }


	function point_radian(p0, p1){
	    return Math.atan2(p0[1] - p1[1], p0[0] - p1[0]);
	}


	function radian_diff(r0, r1) {
	    delta = r0 - r1;
	    sign = (delta < 0? -1.0: 1.0);
	    abs_delta = Math.abs(delta);
	    while(abs_delta >= 2 * Math.PI)
	        abs_delta = abs_delta - 2 * Math.PI;
	    return (abs_delta < - (abs_delta - 2 * Math.PI)? sign * abs_delta : sign * (abs_delta - 2 * Math.PI));
	}

	function get_polar_stat(contour) {
	    sx = 0;
	    sy = 0;
	    len_contour = contour.length;
	    size = Math.max(len_contour, 1)
	    for(var i = 0;i<size;++i) {
	        sx = sx + contour[i][0];
	        sy = sy + contour[i][1];
	    }

	    centroid = [sx / size, sy / size];

	    r = new Float32Array(size);
	    t = new Float32Array(size);

	    for(var i = 0;i<size;++i) {
	        r[i] = point_dist(contour[i], centroid);
	        t[i] = radian_diff(point_radian(contour[(i + 1) % size], centroid), point_radian(contour[i], centroid));
	    }

	    return [r, t];
	}

	const d0 = compute(tf.tensor2d([[0.09, 0.1], [0.1, 0.1], [0.1, 0.11]]));

	window.classify_contour = function(contour_obj, on_inferred_callback) {
	  
	  	var eqi_length = contour_obj.re_contour(2);
  		r_t = get_polar_stat(eqi_length);

  		const input = normalize(r_t[0], r_t[1]);
		const r0 = compute(input);
		const raw = compare(r0, d0);
		const classes = get_class(raw);
		raw.data().then(function(raw_cpu){
			classes.data().then(function(classes_cpu){
	    		on_inferred_callback(contour_obj.id, classes_cpu[0], raw_cpu[0][classes_cpu[0]]);
	    		input.dispose();
	    		r0.dispose();
	    		raw.dispose();
	    		classes.dispose();
			})
		});
	};


};

