
# Torch Roadmap (August 2015 - March 2016)

This roadmap document is intended to serve as a loose plan of our vision for Torch in the short term.  
It is open to community feedback and contribution and only intends to serve as an initial draft.  
After community feedback, we shall freeze it and work on it.  

The roadmap focuses on five separate things

- Core development: improving the core technically. Design changes, code refactors, performance, they go here.
- Documentation and Accessibility: Outlining the changes in documentation, and improving general user and developer documentation in various ways.
- Versioning and Packaging: Planned and much needed changes to the packaging of Torch are discussed here.
- Continuous Build Infrastructure: Making our continuous builds more robust, introducing CUDA and OpenCL contbuilds etc.
- Other improvements


## Torch Core Project Development

 - New class system:
   - **[definite]** with no global side-effects (i.e. the class constructor should be scoped into its parent package)
     Get rid of every statement/system that has a global effect on the environment (torch.setdefaultensortype => dangerous and not clean)
   - **[needs discussion]** fully serializable (i.e. when deserializing/reloading a model, there shouldn't be a need to load libraries that defined the class originally, like nn; the class definition should be serialized as well: this would remove a lot of backward compatibility hacks that we have to add to class definitions currently
       - **koray**: I like this, but wouldn't it break backward compatibility?
		            Currently, whatever we serialize, it is just the data and implementation is defined
					at load time, so if a bug is fixed (or introduced) you use that.
					And it starts being ambiguous, what if I load a layer from file and
					create a new one and their implementation is inconsistent...)
 - **[definite]** Get rid of non-tensor-related stuff (like serialization) in TH, and move it to lua side
 - **[needs discussion]** OpenMP: Should it stay or go? Is Threads sufficient?
       - **Ronan**: I really wonder about this guy, especially now that I have been using threads intensively. I am not sure that fine-grine threading is necessary.
	   - **koray**: I guess you mean with threading, there is no need for OpenMP, but I disagree.
	          Our convolution layer will use multiple threads and then if we run a ReLu over a huge state space, it would become embarrassingly slow.
			  We shouldn't expect everyone to run their experiments in a threading framework. It is more work than necessary sometimes.)
 - **[needs discussion]** Templated C++ in TH Core?
                    - **Ronan**: Should I cleanup TH core? In the end, I am scared to move to C++, but some iterators based taking a closure could be nice (I have some of those that I could add easily).
					         I could move to C++ if it was only template + keeping pointers (and not C++11/14/17, because that would limit the number of users that it can reach because of the latest compilers needed etc.).
 - **[definite]** Migrate to a single, better/modern testing support
              - **koray**: like some aspects of Totem, but should be in core Tester
 - **[definite]** Benchmarking support in Tester
 - **[definite]** Consistent testing scripts across all core projects
 - **[definite]** 'nn' container unified interface between containers and graph
 - **[mostly definite]** Switch to batch only assumption in 'nn'. Right now, the code is unnecessarily complicated for stochastic/batch confusion, we needed extra functions like nInputDims and such.
 - **[needs discussion]** Support named arguments in the constructor for all 'nn' layers.
 - **[definite]** 'rnn' package.
      - **Soumith**: Nicholas Leonard's seems to be a good one.
 - **[mostly definite]** argcheck for all core functions in torch. Get rid of cwrap's ugliness.
 - **[definite]** improve paths to support more file system operations
       - **Clement**: could lfs and penlight be made more standard? penlight is a heavy package but provides so much utility
	   - **Soumith**: I think penlight is lightweight and provides strong utility, definitely consider dependence.
 - **[definite]** JIT/Lua/FFI/GC:
   - **koray**: I think Torch should be agnostic to whatever is the backend;
   - **clement**: yes!
   - at this point, we need to have all core packages use the regular Lua api (almost the case)
     - **Ronan**: agreed.

- **[definite]** plan to have standalone FFI?
  - Facebook releases their puc LUA based FFI package mostly improved by Sam Gross
  - [needs discussion] **Ronan** improves it a bit more to use Leon's C99 parser
                         - **Koray**: I am not opposed to Leon's C99 parser, but we should not have the QT like situation where
						       it relies mostly on Leon to maintain it.
							   And, still we need to have FFI since there are people and packages that rely on it now.
- **[definite]** Lua 5.2 migration (I think it's already finished ;) ).
- **[mostly definite]** Lua 5.3 migration
- **[mostly definite]** Optionally replace GC by Ref-counting (existing version in luajit-rocks; but completely broken but will need to be fixed)
- **[needs discussion]** Make OpenCL support more visible under torch/opencl (**Soumith**: Hugh Perkins will maintain it of course ;) ).
- **[definite]** Split nn into THNN and nn. THNN would be NN package using TH as backend and nn would be the lua layer. THNN can be used as a standalone C library. Same for cunn
- **[Definite]** CUDA typed tensor support - CudaHalfTensor CudaDoubleTensor etc.
- **[Definite]** better plotting support
- **[needs discussion]** UI package that doesn't suck?
  - **Ronan**: something based on cairo?
    - **clement**: not sure if this would have much adoption
    - **Ronan**: yes, it is a worry. I started to do some fancy stuff there, it is not that hard.
	         However, I would need quite some time to polish it.
			 I think having something fully customizable from lua really 
                         makes a difference (rather than something like Qt, for example). 
  - something based on a web client?
      - **clement**: i like the idea of itorch but could never easily build it, build process is too big.
      - **Ronan**: I cannot use something which forces me to use global variables.
      - **koray**: I think at the end of the day, we need to have both a GUI client and a web based client.
		   My main problem with web based clients is that I can't easily create 
                   custom displays to play an animation or such.
		   It is an offline process that I need to generate a movie and then load it in.
		   This and similar things make it hard to use for me.
		   Also, I agree, I actually could not install iTorch on my laptop 
                   before cvpr tutorial somehow, it did not want to work :).
  - **soumith**: I think we should propose a common display API that any interface can implement, 
                 that way the users don't need to change scripts across different UI backends.
	         Also, szym/display is a good candidate for the Web UI, ITorch is indeed a bit of a pain to install.

  - Should we endorse iTorch for everyone to use? 
    - **Ronan**: I know **Soumith** likes it, but I am not a big fan. 
    -            Heavy+encourages the use of global variables. Excellent for tutorials, though.
 	   - This ties to the first question in **Other Questions** section.
 	   - Can we/community do pull requests on iTorch? ( **Soumith**: Yes )
 	   - First step would be to leanify dependencies and/or install procedure (**Soumith**: agreed)
- **[needs discussion]** How about Penlight? It has many crucial things that people use.
   Should we endorse it, use some things from it? Replicate some things in penlight in torch?
   - **clement**: upvoting this! we use it extensively.
   - **Ronan**: I live better with less abstractions, but I can be convinced there.
          However, I find penlight quite big.
          There are things like the classes that I do not like as well (because of the way they chose for creating classes).
- **[needs discussion]** how about Moses? New lean functional package that's pretty useful
- **[definite]** A style guide
  - Guidelines are super important:
    - for Lua: at least impose strict camel case + 3 spaces (no tab)
    - for C: camel case + use of underscore to represent namespace scoping + 2 spaces

## Documentation + Accessibility

 - Tutorials: provide guidelines and basic framework/standard to write and publish tutorials?
 - Universal dataset API
   - Dataset classes for several popular datasets
   - high performance, thread support etc.
   - support CPU and GPU
 - Model Zoo + Training scripts, with training scripts we can highlight Torch's strengths
  - How do we build a super friendly model zoo? git repo of pre-trained models?
    - Better documentation support, have a doc server
 	- Documentation for TH/THC interface and design
 	- Inline documentation parser
 - doc/shell integration (maybe this is still working but needs redoing?)

## Versioning + Packaging
 - Package owners need to start releasing frequent versions (i.e. torch v7.0.1, 7.0.2, ...)
 - scm packages should become deprecated
 - Packages need to avoid global side effects, and return themselves as simple tables (Lua 5.2 started enforcing this on the C side)
 - Provide standard AMI instances that people can launch (already loosely done by the community). We can load it with many standard+optional packages and/or provide one line option to update to latest.

## Build Infrastructure Requirements
 - Prepare core distro release
 - Professional Continuous build for distro and individual core projects
 - Continuous build for GPU
 	- continuous build should include testing
 - The distro should be build and tested at every pull into any of the member projects
 - CI for Linux and OSX

## Other Questions?
 - If there is a project that seems good from outside or consortium, how do we endorse/improve/modify that?
 	- do we put some technical criteria to do that?
 	- being able to do pull requests?
	- Licensing?
 	- or maybe maintain a list of suggested packages?
 	- when does existence of a package stop us from developing the same in core torch?
	- **Soumith**: I think this should largely be community driven and by popularity. Top starred or watched repos in the ecosystem would be a good start.
 	
