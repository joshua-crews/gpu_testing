use cudarc::{
    driver::{CudaContext, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};

pub fn add(a_host: Vec<i32>, b_host: Vec<i32>) -> Result<Vec<i32>, Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let module = ctx.load_module(Ptx::from_file("kernel.ptx"))?;
    let func = module.load_function("add").unwrap();

    let a = stream.memcpy_stod(&a_host)?;
    let b = stream.memcpy_stod(&b_host)?;
    let mut c = stream.alloc_zeros::<i32>(4)?;

    let cfg = LaunchConfig::for_num_elems(a_host.len().try_into().unwrap());
    let mut launch = stream.launch_builder(&func);
    launch.arg(&a);
    launch.arg(&b);
    launch.arg(&mut c);
    unsafe { launch.launch(cfg)?; }

    Ok(stream.memcpy_dtov(&c)?)
}

pub fn multiply(a_host: Vec<i32>, b_host: Vec<i32>) -> Result<Vec<i32>, Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let module = ctx.load_module(Ptx::from_file("kernel.ptx"))?;
    let func = module.load_function("multiply").unwrap();

    let a = stream.memcpy_stod(&a_host)?;
    let b = stream.memcpy_stod(&b_host)?;
    let mut c = stream.alloc_zeros::<i32>(4)?;

    let cfg = LaunchConfig::for_num_elems(a_host.len().try_into().unwrap());
    let mut launch = stream.launch_builder(&func);
    launch.arg(&a);
    launch.arg(&b);
    launch.arg(&mut c);
    unsafe { launch.launch(cfg)?; }

    Ok(stream.memcpy_dtov(&c)?)
}

pub fn dot_product(a_host: Vec<i32>, b_host: Vec<i32>) -> Result<i32, Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let module = ctx.load_module(Ptx::from_file("kernel.ptx"))?;
    let func = module.load_function("dot_product")?;

    let a = stream.memcpy_stod(&a_host)?;
    let b = stream.memcpy_stod(&b_host)?;
    let mut c = stream.alloc_zeros::<i32>(1)?;

    let threads_per_block = 256;
    let cfg = LaunchConfig {
        grid_dim: (u32::div_ceil(a_host.len() as u32, threads_per_block), 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: threads_per_block * std::mem::size_of::<i32>() as u32,
    };

    let mut launch = stream.launch_builder(&func);
    launch.arg(&a);
    launch.arg(&b);
    launch.arg(&mut c);
    let binding = a_host.len() as i32;
    launch.arg(&binding);
    unsafe { launch.launch(cfg)?; }

    let result = stream.memcpy_dtov(&c)?;
    Ok(result[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let a_host = vec![1, 2, 3, 4];
        let b_host = vec![10, 20, 30, 40];
        let result = add(a_host, b_host).unwrap();
        assert_eq!(result, vec![11, 22, 33, 44]);
    }

    #[test]
    fn test_multiply() {
        let a_host = vec![1, 2, 3, 4];
        let b_host = vec![1, 3, 5, 7];
        let result = multiply(a_host, b_host).unwrap();
        assert_eq!(result, vec![1, 6, 15, 28]);
    }

    #[test]
    fn test_dot_product() {
        for _ in 1..101 {
            let a_host = vec![1, 3, -5];
            let b_host = vec![4, -2, -1];
            let result = dot_product(a_host, b_host).unwrap();
            assert_eq!(result, 3);
        }
    }
}
