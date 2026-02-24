# Large-scale model training on Chameleon - single GPU

In this tutorial, we will practice fine-tuning a large language model. We will use a selection of techniques to allow us to train models that would not otherwise fit in GPU memory:

-   gradient accumulation
-   reduced precision
-   parameter efficient fine tuning

To run this experiment, you should have already created an account on Chameleon, and become part of a project.

You must also have added your SSH key to the CHI@UC site (to use an A100 GPU) or KVM@TACC site (to use an H100 GPU).

## Experiment topology

In this experiment, we will deploy a single instance with a GPU. We have "tuned" this experiment for two specific GPU types:

-   an A100 with 80 GB VRAM (available on most `compute_gigaio` bare metal instances at CHI@UC)
-   or an H100 with 94GB VRAM (available in the `g1.h100.pci.1` flavor at KVM@TACC)

(Generally, to find a Chameleon node with a specific GPU type, we can use the Chameleon [Hardware Browser](https://chameleoncloud.org/hardware/). )

You are currently viewing the A100 version of the instructions, but H100 instructions are also available at [index_h100](index_h100).

## Create a lease

To use a GPU instance on Chameleon, we must reserve it in advance. GPU instances are much more in-demand than other resource types, and so we typically cannot make a reservation "on the spot" to use one.

We can use the OpenStack graphical user interface, Horizon, to reserve a GPU in advance. To access this interface,

-   from the [Chameleon website](https://chameleoncloud.org/hardware/)
-   click "Experiment" \> "CHI@UC"
-   log in if prompted to do so
-   check the project drop-down menu near the top left (which shows e.g. "CHI-XXXXXX"), and make sure the correct project is selected.

Reserve a 2 hr 50 minute block on a node with a single A100 80GB GPU. We will use `compute_gigaio`.

-   On the left side, click on "Reservations" \> "Leases", and then click on "Host Calendar". In the "Node type" drop down menu, change the type to `compute_gigaio` to see the schedule of availability. You may change the date range setting to "30 days" to see a longer time scale. Note that the dates and times in this display are in UTC, so you will need to convert to your local time zone.
-   Once you have identified an available 2 hr 50 minute block in UTC time that works for you in your local time zone, make a note of:
    -   the start and end time of the time you will try to reserve. (Note that if you mouse over an existing reservation, a pop up will show you the exact start and end time of that reservation.)
    -   and the name of the node you want to reserve.
-   Then, on the left side, click on the name of the node you want to reserve:
    -   set the "Name" to `llm_single_netID`, replacing `netID` with your actual net ID.
    -   set the start date and time in UTC
    -   modify the lease length (in days) until the end date is correct. Then, set the end time. To be mindful of other users, you should limit your lease time as directed.
    -   Click "Next".
-   On the "Hosts" tab, confirm that the node you selected is listed in the "Resource properties" section, and click "Next".
-   Then, click "Create". (We won't include any network resources in this lease.)

Your lease status should show as "Pending". If you click on the lease, you can see an overview, including the start time and end time, and it will show the name of the physical host that is reserved for you as part of your lease.

At the beginning of your lease time, continue with `2_create_server.ipynb`.

# Large-scale model training on Chameleon - multi GPU

In this tutorial, we will practice fine-tuning a large language model. We will try two different strategies that distribute training across multiple GPUs:

-   DDP
-   FSDP

To run this experiment, you should have already created an account on Chameleon, and become part of a project.

You must also have added your SSH key to the CHI@UC site (to use a 4x A100 GPU) or KVM@TACC site (to use a 4x H100 GPU).

## Experiment topology

In this experiment, we will deploy a single instance with four GPUs. We have "tuned" this experiment for two specific GPU types:

-   4x A100 with 80 GB VRAM (available on most `gpu_a100_pcie` bare metal instances at CHI@UC)
-   or 4x H100 with 94GB VRAM (available in the `g1.h100.pci.4` flavor at KVM@TACC)

(Generally, to find a Chameleon node with a specific GPU type, we can use the Chameleon [Hardware Browser](https://chameleoncloud.org/hardware/). )

You are currently viewing the A100 version of the instructions, but H100 instructions are also available at [index_h100](index_h100).

## Create a lease

To use a GPU instance on Chameleon, we must reserve it in advance. GPU instances are much more in-demand than other resource types, and so we typically cannot make a reservation "on the spot" to use one.

We can use the OpenStack graphical user interface, Horizon, to reserve a GPU in advance. To access this interface,

-   from the [Chameleon website](https://chameleoncloud.org/hardware/)
-   click "Experiment" \> "CHI@UC"
-   log in if prompted to do so
-   check the project drop-down menu near the top left (which shows e.g. "CHI-XXXXXX"), and make sure the correct project is selected.

Reserve a 2 hr 50 minute block on a node with four A100 80GBs GPU: `gpu_a100_pcie`.

-   On the left side, click on "Reservations" \> "Leases", and then click on "Host Calendar". In the "Node type" drop down menu, change the type to `gpu_a100_pcie` to see the schedule of availability. You may change the date range setting to "30 days" to see a longer time scale. Note that the dates and times in this display are in UTC, so you will need to convert to your local time zone.
-   Once you have identified an available 2 hr 50 minute block in UTC time that works for you in your local time zone, make a note of:
    -   the start and end time of the time you will try to reserve. (Note that if you mouse over an existing reservation, a pop up will show you the exact start and end time of that reservation.)
    -   and the name of the node you want to reserve.
-   Then, on the left side, click on the name of the node you want to reserve:
    -   set the "Name" to `llm_multi_netID`, replacing `netID` with your actual net ID.
    -   set the start date and time in UTC
    -   modify the lease length (in days) until the end date is correct. Then, set the end time. To be mindful of other users, you should limit your lease time as directed.
    -   Click "Next".
-   On the "Hosts" tab, confirm that the node you selected is listed in the "Resource properties" section, and click "Next".
-   Then, click "Create". (We won't include any network resources in this lease.)

Your lease status should show as "Pending". If you click on the lease, you can see an overview, including the start time and end time, and it will show the name of the physical host that is reserved for you as part of your lease.

At the beginning of your lease time, continue with `2_create_server.ipynb`.
