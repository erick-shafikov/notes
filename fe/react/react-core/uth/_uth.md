Порядок вызовов внутренних функций React, по стадиям

# Стадия виртуального dom

На этой стадии формируется виртуальный dom

Ниже callstack функций:
- root
- legacyRenderSubtreeContainer (неконкурентный режим)
- - legacyCreateRootFromDOMContainer
- unbatchedUpdates
- - executionContext
- updateContainer
- - requestUpdateLane
- - - lane
- - enqueueUpdate
- - fiber
- - - fiber.child
- - entangleTransition
- scheduleUpdateOnFiber
- - workInProgressRoot
- - fiberNode

# Стадия scheduled renders

Callstack:
- performSyncWorkOnRoot
- - flushPassiveEffects
- renderRootSync
- workLoopSync
- - performUnitOfWork
- - workInProgress
- - workLoopSync (цикл)
- - - workInProgress.alternate
- - - workInProgress.tag
- - - workInProgress.child
- - - reconcileChildren
- - - placeSingleChild -> FiberNode

# Стадия commit

CallStack:
- commitRoot
- - flushPassiveEffects
- - workInProgressRoot
- commitMutationEffects
- commitMutationEffectsOnFiber
- commitPlacement
- - finishedWork.flags
- - stateNode
- insertOrAppendPlacementNodeIntoContainer


